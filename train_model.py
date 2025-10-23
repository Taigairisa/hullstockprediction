import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

def create_dataset(X, y, time_steps=1):
    """
    LSTM用のデータセットを作成する関数
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_and_train_lstm():
    """
    データの読み込み、前処理、LSTMモデルの構築、学習、評価を行うメイン関数
    """
    # --- GPUの利用状況を確認 ---
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 現在は、単一GPUのみを想定
            # 必要に応じて複数のGPUを利用する設定も可能です
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # 可視デバイスはプログラムの開始時に設定する必要があるため、エラーになることがあります
            print(e)
    else:
        print("GPUが見つかりません。CPUで実行します。")

    # --- 1. データの読み込みと前処理 ---
    try:
        df = pd.read_csv('/home/administrator/shared/repos/hullstockprediction/data/train.csv')
    except FileNotFoundError:
        print("エラー: train.csvが見つかりません。ファイルパスを確認してください。")
        return

    # 予測対象の列名を確認（CSVヘッダーに合わせる）
    target_col = 'market_forward_excess_returns'
    if target_col not in df.columns:
        print(f"エラー: ターゲット列 '{target_col}' がCSVファイルに存在しません。")
        return
        
    # 除外する列のリストを作成
    # 'D'で始まる列を動的に取得
    cols_to_drop = [col for col in df.columns if col.startswith('D')]
    # 予測対象とその他の指定された列を追加
    cols_to_drop.extend([target_col, 'forward_returns', 'risk_free_rate', 'date_id'])

    # 特徴量(X)とターゲット(y)を定義
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # 欠損値を0で埋める
    X = X.fillna(0)
    y = y.fillna(0)

    # データの正規化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # --- 2. LSTM用データセットの作成 ---
    TIME_STEPS = 60
    X_data, y_data = create_dataset(X_scaled, y_scaled, TIME_STEPS)

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=False)

    print(f"訓練データの形状: {X_train.shape}")
    print(f"テストデータの形状: {X_test.shape}")

    # --- 3. LSTMモデルの構築 ---
    model = Sequential()
    # 入力層 + LSTM層1
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    # LSTM層2
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # 全結合層
    model.add(Dense(units=25))
    # 出力層
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # --- 4. モデルの学習 ---
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # --- 5. 予測と評価 ---
    predictions_scaled = model.predict(X_test)
    # スケールを元に戻す
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # 評価指標 (MSE)
    mse = np.mean((predictions - y_test_orig)**2)
    print(f"\nテストデータのMean Squared Error: {mse}")

    # --- 6. 結果の可視化 ---
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_orig, color='blue', label='Actual Market Excess Return')
    plt.plot(predictions, color='red', label='Predicted Market Excess Return')
    plt.title('Stock Market Excess Return Prediction')
    plt.xlabel('Time')
    plt.ylabel('Market Excess Return')
    plt.legend()
    
    # 保存先ディレクトリを作成
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    print(f"\n予測結果のグラフを '{output_dir}/prediction_vs_actual.png' に保存しました。")
    # plt.show() # GUI環境で実行している場合はコメントを外してください

if __name__ == '__main__':
    build_and_train_lstm()
