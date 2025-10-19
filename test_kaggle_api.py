import os
from pathlib import Path
import pandas as pd
import polars as pl
import kaggle_evaluation.default_inference_server

def predict(test: pl.DataFrame):
    """テストデータを受け取り、予測ベクトルを返す"""
    # ダミー予測（すべて 0.0）
    n = len(test)
    preds = pl.Series([0.0] * n, dtype=pl.Float64)
    print(f"Received batch with {len(test)} rows")
    # ★ DataFrame で返す（列名は target 名）
    return pl.DataFrame({"market_forward_excess_returns": preds})


inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" 

    if not (data_dir / "test.csv").exists():
        raise FileNotFoundError(f"test.csv が見つかりません: {data_dir}")

    # run_local_gateway に文字列パスを渡す
    inference_server.run_local_gateway((str(data_dir),))
