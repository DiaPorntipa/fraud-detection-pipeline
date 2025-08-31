import pandas as pd
import numpy as np
from pathlib import Path
from features import make_time_feats, integrity_flags

IN_PATH = Path("data/fraud_mock.csv")
OUT_PATH = Path("data/processed.csv")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = make_time_feats(df)
    df = integrity_flags(df)

    df["amount_over_src"] = df["amount"] / (df["src_bal"].replace(0, np.nan))
    df["amount_over_src"] = df["amount_over_src"].fillna(0).round(4)

    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "transac_type", "amount", "src_bal", "dst_bal",
        "hour", "day_of_week", "flag_any_inconsistency", "amount_over_src",
        "is_fraud", "is_flagged_fraud"
    ]

    return df[keep]


def main():
    # TODO (LOW): Add an optional --from-cache to load a preprocessed file, but default = raw
    df = pd.read_csv(IN_PATH)

    # Basic sanity: keep non-negative numeric values only
    num_cols = df.select_dtypes(include="number").columns
    df = df[(df[num_cols] >= 0).all(axis=1)].copy()

    df = feature_engineering(df)
    df = select_columns(df)

    # Save data
    # TODO (LOW): Consider using Parquet for speed
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved data: {OUT_PATH}  ({len(df):,} rows)")

    # Show fraud rate in data
    rate = 100 * df["is_fraud"].mean()
    print(f"Fraud rate in data: {rate:.4f}%")


if __name__ == "__main__":
    main()
