# main.py
import os
import pandas as pd
import numpy as np

from utils.data_fetcher import DataFetcher

# 状态识别
from new.regime_hmm import compute_states_from_df

# 策略
from strategies.ma_strategy import MovingAverageStrategy
from strategies.state_driven_strategy import StateDrivenStrategy
from strategies.meta_model_strategy import MetaModelStrategy
from strategies.rag_enhanced_strategy import RAGEnhancedStrategy

from backtest.backtester import Backtester


# =========================
# 全局配置
# =========================
PRICE_FILENAME = "601788_SH.csv"

# 目标个股自己的 embeddings 文件（不是 sector memory）
EMBEDDINGS_PATH = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\embeddings_ht_qt.csv"

# 你按截止日期 2021-12-21 构建好的 RAG 经验库目录
RAG_MEMORY_DIR = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\new\memory\broker_cut_20211231"

# 回测截止日期：只允许回测 / 使用到这一天
BACKTEST_CUTOFF_DATE = "2021-12-31"


# =========================
# 数据读取
# =========================
def _load_price_data(fetcher: DataFetcher, filename: str) -> pd.DataFrame:
    """
    读取行情CSV（DataFetcher 会自动加 data/ 前缀）
    返回：index=trade_date
    """
    df = fetcher.read_data_from_csv(filename)

    if df.index.name is None:
        df.index.name = "trade_date"

    # 强制索引为 datetime
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.sort_index()

    return df


def _load_embeddings(path: str) -> pd.DataFrame:
    """
    读取目标个股 embeddings_ht_qt.csv（含 qt_* / y_hat / t_index 等）
    期望有 trade_date 或 date 字段。
    """
    emb = pd.read_csv(path, encoding="utf-8-sig")

    if "trade_date" not in emb.columns and "date" in emb.columns:
        emb = emb.rename(columns={"date": "trade_date"})

    if "trade_date" not in emb.columns:
        raise ValueError(f"embeddings 缺少 trade_date/date 列。现有列：{list(emb.columns)[:50]}")

    emb["trade_date"] = pd.to_datetime(emb["trade_date"], errors="coerce")
    emb = emb.dropna(subset=["trade_date"]).copy()
    emb = emb.sort_values("trade_date").reset_index(drop=True)

    return emb


def _cut_price_by_date(price_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    cutoff_ts = pd.to_datetime(cutoff_date)
    out = price_df.loc[price_df.index <= cutoff_ts].copy()
    if len(out) == 0:
        raise ValueError(f"price_df 裁剪后为空，请检查 cutoff_date={cutoff_date}")
    return out


def _cut_embeddings_by_date(emb_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    cutoff_ts = pd.to_datetime(cutoff_date)
    out = emb_df[emb_df["trade_date"] <= cutoff_ts].copy()
    out = out.sort_values("trade_date").reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"embeddings 裁剪后为空，请检查 cutoff_date={cutoff_date}")
    return out


def _merge_price_and_embeddings(price_df: pd.DataFrame, emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    强制把两边 trade_date 都转成 'YYYY-MM-DD' 字符串再 merge，避免 dtype 不一致。
    """
    price_reset = price_df.reset_index()
    price_reset = price_reset.rename(columns={price_reset.columns[0]: "trade_date"})
    price_reset["trade_date"] = pd.to_datetime(price_reset["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "trade_date" not in emb_df.columns:
        raise ValueError(f"embeddings 缺少 trade_date 列。现有列：{list(emb_df.columns)[:50]}")

    emb = emb_df.copy()
    emb["trade_date"] = pd.to_datetime(emb["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    price_reset = price_reset.dropna(subset=["trade_date"])
    emb = emb.dropna(subset=["trade_date"])

    merged = price_reset.merge(emb, on="trade_date", how="inner")
    merged = merged.set_index("trade_date")
    merged.index.name = "trade_date"

    # 再转回 datetime index
    merged.index = pd.to_datetime(merged.index, errors="coerce")
    merged = merged[~merged.index.isna()].copy()
    merged = merged.sort_index()

    return merged


# =========================
# 状态特征
# =========================
def _add_market_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    调用 compute_states_from_df 增加市场状态列。
    在进入 HMM 前强制保证 open/high/low/close 存在
    """
    print("计算市场状态...")

    df = df.copy()

    def _pick(src_cols, preferred):
        for c in preferred:
            if c in src_cols:
                return c
        return None

    cols = list(df.columns)

    close_col = _pick(cols, ["close", "close_x", "close_y", "Close", "adj_close", "close_price", "收盘"])
    open_col  = _pick(cols, ["open", "open_x", "open_y", "Open", "开盘"])
    high_col  = _pick(cols, ["high", "high_x", "high_y", "High", "最高"])
    low_col   = _pick(cols, ["low", "low_x", "low_y", "Low", "最低"])

    if close_col is None:
        raise ValueError(f"[HMM] 找不到 close 列。现有列：{cols}")

    if open_col is None or high_col is None or low_col is None:
        raise ValueError(f"[HMM] 找不到完整 OHLC(open/high/low)。现有列：{cols}")

    if close_col != "close":
        df["close"] = df[close_col]
    if open_col != "open":
        df["open"] = df[open_col]
    if high_col != "high":
        df["high"] = df[high_col]
    if low_col != "low":
        df["low"] = df[low_col]

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})
    df_reset["date"] = pd.to_datetime(df_reset["date"], errors="coerce")
    df_reset = df_reset.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)

    print("[DEBUG] HMM input columns:", df_reset.columns.tolist())
    print(df_reset[["date", "open", "high", "low", "close"]].head())

    df_with_states = compute_states_from_df(df_reset)

    df_with_states = df_with_states.set_index("date")
    df_with_states.index.name = "trade_date"
    df_with_states = df_with_states.sort_index()

    return df_with_states


def _standardize_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)

    if "vol" not in cols:
        if "vol_x" in cols:
            df["vol"] = df["vol_x"]
        elif "vol_y" in cols:
            df["vol"] = df["vol_y"]
        elif "volume" in cols:
            df["vol"] = df["volume"]
        elif "Volume" in cols:
            df["vol"] = df["Volume"]
        else:
            df["vol"] = 0.0

    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)
    return df


# =========================
# 主流程
# =========================
def main():
    # ========== 1. 加载行情数据 ==========
    fetcher = DataFetcher()
    price_df = _load_price_data(fetcher, PRICE_FILENAME)

    # 只保留到 2021-12-31
    price_df = _cut_price_by_date(price_df, BACKTEST_CUTOFF_DATE)

    # ========== 2. 加载目标个股 embeddings 并裁剪 ==========
    emb_df = _load_embeddings(EMBEDDINGS_PATH)
    emb_df = _cut_embeddings_by_date(emb_df, BACKTEST_CUTOFF_DATE)

    print("合并行情数据与 embeddings 特征...")
    df = _merge_price_and_embeddings(price_df, emb_df)

    if len(df) == 0:
        raise ValueError("price 与 embeddings merge 后为空，请检查 trade_date 对齐情况。")

    # ========== 3. 添加市场状态 ==========
    df_with_states = _add_market_states(df)
    df_with_states = _standardize_volume(df_with_states)

    # 再保险：排序 + 截断
    df_with_states = df_with_states.sort_index()
    df_with_states = df_with_states.loc[df_with_states.index <= pd.to_datetime(BACKTEST_CUTOFF_DATE)].copy()

    # 给回测和 RAG 使用的位置索引
    df_with_states["t_index"] = np.arange(len(df_with_states), dtype=int)

    print("=" * 80)
    print("[INFO] backtest data range:")
    print("start:", df_with_states.index.min())
    print("end  :", df_with_states.index.max())
    print("rows :", len(df_with_states))

    # ========== 4. 定义策略组合 ==========
    strategies = {
        "MA": MovingAverageStrategy(short_window=5, long_window=20),

        "StateDriven_NoPred": StateDrivenStrategy(use_prediction=False),

        "StateDriven_WithPred": StateDrivenStrategy(use_prediction=True),

        "MetaModel": MetaModelStrategy(
            lstm_path='utils/models/lstm_model.h5',
            hmm_path='utils/models/hmm_model.pkl',
            transformer_path='utils/models/transformer_model_state_dict.pt',
            lookback=60,
            selection_mode='three_models',
            pred_kind="logret",
        ),

        # Transformer + RAG 融合策略
        "RAGEnhanced": RAGEnhancedStrategy(
            memory_dir=RAG_MEMORY_DIR,   # 改成截止到 2021-12-31 的安全版 memory
            topk=10,                     
            min_gap=3,                   # 先放宽，避免 early period 没候选
            alpha=0.5,
            buy_th=0.55,
            sell_th=0.45,
            sl_mult=2.0,
            tp_mult=3.0,
            sl_floor=0.02,
            tp_floor=0.03,
            sl_cap=0.12,
            tp_cap=0.20,
            strict_past_only=True,       # 只允许 past experience
            exclude_self=False,          # 先设 False 跑通；稳定后可改 True
        ),
    }

    # ========== 5. 运行回测 ==========
    print("开始回测...")
    backtester = Backtester(
        strategies=strategies,
        data=df_with_states,
        initial_capital=100000.0,
    )
    backtester.run_backtest()


if __name__ == "__main__":
    main()

