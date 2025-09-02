import pandas as pd
import numpy as np


def make_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["time_ind"] % 24
    df["day_of_week"] = (df["time_ind"] // 24) % 7

    return df


def integrity_flags(df: pd.DataFrame, atol=1e-6) -> pd.DataFrame:
    # TODO: Consider dropping the line below for better memory usage
    df = df.copy()

    overdraft = df["amount"] > (df["src_bal"] + atol)
    src_mis = ~np.isclose(df["src_bal"] - df["amount"],
                          df["src_new_bal"], atol=atol)

    dst_missing = (
        df["dst_bal"].isna() | df["dst_new_bal"].isna() |
        ((df["dst_bal"] == 0.0) & (df["dst_new_bal"] == 0.0))
    )
    dst_mis = ~dst_missing & ~np.isclose(
        df["dst_bal"] + df["amount"], df["dst_new_bal"], atol=atol)

    df["flag_overdraft"] = overdraft.astype(int)
    df["flag_src_mismatch"] = src_mis.astype(int)
    df["flag_dst_mismatch"] = dst_mis.astype(int)
    df["flag_any_inconsistency"] = (
        df["flag_overdraft"] | df["flag_src_mismatch"] | df["flag_dst_mismatch"]
    ).astype(int)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = make_time_feats(df)
    df = integrity_flags(df)

    df["amount_over_src"] = df["amount"] / (df["src_bal"].replace(0, np.nan))
    df["amount_over_src"] = df["amount_over_src"].fillna(0).round(4)

    return df
