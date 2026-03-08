import pandas as pd

INPUT_CSV = r"data\601066_SHE.csv"
OUTPUT_CSV = r"data\601066_SHE_standardized.csv"


def normalize_ts_code(x):
    x = str(x).strip().upper()

    if x.endswith("-SHE"):
        return x.replace("-SHE", ".SH")
    if x.endswith("-SZE"):
        return x.replace("-SZE", ".SZ")
    if x.endswith("-BJ"):
        return x.replace("-BJ", ".BJ")

    if x.endswith(".SH") or x.endswith(".SZ") or x.endswith(".BJ"):
        return x

    if x.isdigit():
        if x.startswith(("6", "9")):
            return f"{x}.SH"
        if x.startswith(("0", "2", "3")):
            return f"{x}.SZ"
        if x.startswith(("4", "8")):
            return f"{x}.BJ"

    return x


def standardize_601066():
    # 关键修改：加 index_col=False，避免首列错位
    df = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig",
        engine="python",
        index_col=False
    )

    # 删除全空列
    df = df.dropna(axis=1, how="all")

    rename_map = {
        "证券代码": "ts_code",
        "交易日期": "trade_date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交数量(股)": "vol",
    }
    df = df.rename(columns=rename_map)

    keep_cols = ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
    df = df[keep_cols].copy()

    # ts_code 修正
    df["ts_code"] = df["ts_code"].apply(normalize_ts_code)

    # trade_date 修正
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"])
    df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")

    # 数值列
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

    # 排序
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 保存
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    standardize_601066()
