import os
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "raw"
PARQUET_PATH = CACHE_DIR / "tiktok_yt.parquet"


def get_dataframe(force_download: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_download and PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)

    try:
        from datasets import load_dataset
        df = load_dataset("tarekmasryo/youtube-tiktok-trends-dataset-2025")["train"].to_pandas()
        df.to_parquet(PARQUET_PATH, index=False)
        return df
    except Exception as e:
        if PARQUET_PATH.exists():
            return pd.read_parquet(PARQUET_PATH)
        raise RuntimeError(f"Falha ao baixar dataset: {e}")


if __name__ == "__main__":
    df = get_dataframe()
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
