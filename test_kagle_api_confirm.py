import polars as pl

sub = pl.read_parquet("submission.parquet")
print(sub.head())
print(sub.shape)
