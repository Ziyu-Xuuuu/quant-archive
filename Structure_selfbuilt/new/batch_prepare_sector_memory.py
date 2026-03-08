import os
import sys
import subprocess
import pandas as pd
import numpy as np

from regime_hmm import compute_states_from_df


BASE_DIR = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt"
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = DATA_DIR
PROCESSED_DIR = DATA_DIR

MERGED_OUTPUT = os.path.join(DATA_DIR, "sector_embeddings_memory.csv")

MODEL_PATH = os.path.join(BASE_DIR, "new", "transformer_model.keras")
GENERATE_EMBED_SCRIPT = os.path.join(BASE_DIR, "new", "Generate_embeddings.py")

H = 5
WINDOW = 30
SECTOR_NAME = "broker"

STOCK_FILES = [
    ("002736_SZSE_standardized.csv", "002736.SZ"),
    ("600999_SHE_standardized.csv", "600999.SH"),
    ("601066_SHE_standardized.csv", "601066.SH"),
    ("601995_SHE_standardized.csv", "601995.SH"),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_yyyymmdd_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"trade_date": str, "date": str})

    if "trade_date" not in df.columns and "date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "trade_date"})

    if "trade_date" in df.columns:
        df["trade_date_raw"] = df["trade_date"].astype(str).str.strip()
        df["trade_date"] = parse_yyyymmdd_series(df["trade_date_raw"])
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    elif "date" in df.columns:
        df["trade_date_raw"] = df["date"].astype(str).str.strip()
        df["date"] = parse_yyyymmdd_series(df["trade_date_raw"])
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def normalize_columns(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    df = df.copy()

    date_candidates = [c for c in ["date", "trade_date", "交易日期"] if c in df.columns]
    if not date_candidates:
        first_col = df.columns[0]
        date_col = first_col
    else:
        date_col = date_candidates[0]

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    for extra_date_col in ["trade_date", "交易日期"]:
        if extra_date_col in df.columns and extra_date_col != date_col:
            df = df.drop(columns=[extra_date_col])

    rename_map = {
        "证券代码": "ts_code",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "vol",
        "成交数量(股)": "vol",
        "volume": "vol",
    }
    df = df.rename(columns=rename_map)

    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        df = df.loc[:, ~df.columns.duplicated()]

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ts_code} 缺少必要列: {missing}，当前列为: {df.columns.tolist()}")

    if "vol" not in df.columns:
        df["vol"] = np.nan

    if "ts_code" not in df.columns:
        df["ts_code"] = ts_code
    else:
        df["ts_code"] = df["ts_code"].fillna(ts_code).astype(str)

    if "trade_date_raw" not in df.columns:
        df["trade_date_raw"] = df["date"].astype(str).str.strip()

    df["date"] = parse_yyyymmdd_series(df["trade_date_raw"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def clean_price_data(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    df = df.copy()

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")

    bad_mask = df[price_cols].isna().any(axis=1)
    bad_mask |= (df[price_cols] <= 0).any(axis=1)

    df = df.loc[~bad_mask].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{ts_code} 清洗后无有效数据。")

    return df


def make_with_labels(df_states: pd.DataFrame, h: int = 5) -> pd.DataFrame:
    df = df_states.copy()

    if "date" in df.columns and "trade_date" not in df.columns:
        df = df.rename(columns={"date": "trade_date"})

    if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df["trade_date"] = parse_yyyymmdd_series(df["trade_date"])

    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"] > 0].copy()

    df["fwd_ret"] = np.log(df["close"]).shift(-h) - np.log(df["close"])
    df["y"] = (df["fwd_ret"] > 0).astype(int)

    df = df.dropna(subset=["fwd_ret"]).reset_index(drop=True)

    return df


def format_with_labels_output(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    df = df.copy()

    if "ts_code" not in df.columns:
        df["ts_code"] = ts_code
    else:
        df["ts_code"] = df["ts_code"].fillna(ts_code)

    required_output_cols = [
        "trade_date", "ts_code", "open", "high", "low", "close", "vol",
        "trend_regime", "vol_regime", "state", "y",
        "hmm_state", "hmm_state_label",
        "hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"
    ]

    for col in required_output_cols:
        if col not in df.columns:
            df[col] = np.nan

    if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df["trade_date"] = pd.to_datetime(
            df["trade_date"].astype(str).str.strip(),
            format="%Y-%m-%d",
            errors="coerce"
        )

    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    # 只转换真正的数值列
    numeric_cols = [
        "open", "high", "low", "close", "vol",
        "y", "hmm_state",
        "hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 字符串列不要转 numeric
    string_cols = ["ts_code", "trend_regime", "vol_regime", "state", "hmm_state_label"]
    for col in string_cols:
        df[col] = df[col].where(df[col].notna(), pd.NA)

    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")

    return df[required_output_cols].copy()

def run_generate_embeddings(input_csv: str, output_csv: str) -> None:
    cmd = [
        sys.executable,
        GENERATE_EMBED_SCRIPT,
        "--input", input_csv,
        "--model", MODEL_PATH,
        "--output", output_csv,
        "--window", str(WINDOW),
        "--ht_layer", "ht",
        "--qt_layer", "qt",
        "--yhat_layer", "y_hat",
    ]
    subprocess.run(cmd, check=True)


def main():
    ensure_dir(PROCESSED_DIR)

    embedding_files = []

    for raw_file, ts_code in STOCK_FILES:
        print("=" * 80)
        print(f"[PROCESS] {ts_code} | raw={raw_file}")

        raw_path = os.path.join(RAW_DATA_DIR, raw_file)
        if not os.path.exists(raw_path):
            print(f"[SKIP] 文件不存在: {raw_path}")
            continue

        df_raw = load_raw_csv(raw_path)
        df_norm = normalize_columns(df_raw, ts_code=ts_code)
        df_norm = clean_price_data(df_norm, ts_code=ts_code)

        if len(df_norm) < 60:
            print(f"[SKIP] {ts_code} 清洗后样本太少: {len(df_norm)}")
            continue

        df_states = compute_states_from_df(df_norm)

        if "date" in df_states.columns and "trade_date_raw" in df_norm.columns:
            raw_date_map = df_norm[["date", "trade_date_raw"]].drop_duplicates(subset=["date"])
            df_states = df_states.merge(raw_date_map, on="date", how="left")

        df_with_labels = make_with_labels(df_states, h=H)

        if "trade_date_raw" in df_with_labels.columns:
            mask = df_with_labels["trade_date_raw"].notna()
            df_with_labels.loc[mask, "trade_date"] = pd.to_datetime(
                df_with_labels.loc[mask, "trade_date_raw"].astype(str).str.strip(),
                format="%Y%m%d",
                errors="coerce"
            )

        df_with_labels = format_with_labels_output(df_with_labels, ts_code=ts_code)

        with_labels_name = f"{ts_code.replace('.', '_')}_market_states_with_labels.csv"
        with_labels_path = os.path.join(PROCESSED_DIR, with_labels_name)
        df_with_labels.to_csv(with_labels_path, index=False, encoding="utf-8-sig")
        print(f"[OK] saved with_labels: {with_labels_path}")

        embeddings_name = f"{ts_code.replace('.', '_')}_embeddings_ht_qt.csv"
        embeddings_path = os.path.join(PROCESSED_DIR, embeddings_name)

        run_generate_embeddings(with_labels_path, embeddings_path)
        embedding_files.append((embeddings_path, ts_code))

    if len(embedding_files) == 0:
        raise RuntimeError("没有成功生成任何 embeddings 文件。")

    merged = []
    for emb_path, ts_code in embedding_files:
        df = pd.read_csv(emb_path, encoding="utf-8-sig")

        if "ts_code" not in df.columns:
            df["ts_code"] = ts_code

        df["source_stock"] = ts_code
        df["sector_name"] = SECTOR_NAME
        merged.append(df)

    df_merged = pd.concat(merged, axis=0, ignore_index=True)
    df_merged.to_csv(MERGED_OUTPUT, index=False, encoding="utf-8-sig")

    print(MERGED_OUTPUT)
    print(df_merged.shape)


if __name__ == "__main__":
    main()

