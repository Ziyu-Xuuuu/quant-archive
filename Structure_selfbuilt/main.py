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
from strategies.rag_enhanced_strategy import RAGEnhancedStrategy  #  新增

from backtest.backtester import Backtester


def _load_price_data(fetcher: DataFetcher, filename: str) -> pd.DataFrame:
    """
    读取行情CSV（ DataFetcher 会自动加 data/ 前缀）
    返回：index=trade_date
    """
    df = fetcher.read_data_from_csv(filename)
    # 确保索引名一致
    if df.index.name is None:
        df.index.name = "trade_date"
    return df


def _load_embeddings(path: str) -> pd.DataFrame:
    """
    读取 embeddings_ht_qt.csv（含 qt_* / y_hat / t_index 等）
    期望有 trade_date 或 date 字段。
    """
    emb = pd.read_csv(path, encoding="utf-8-sig")
    if "trade_date" not in emb.columns and "date" in emb.columns:
        emb = emb.rename(columns={"date": "trade_date"})
    return emb


def _merge_price_and_embeddings(price_df: pd.DataFrame, emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    强制把两边 trade_date 都转成 'YYYY-MM-DD' 字符串再 merge，避免 dtype 不一致。
    """
    # price_df: index=trade_date
    price_reset = price_df.reset_index()
    price_reset = price_reset.rename(columns={price_reset.columns[0]: "trade_date"})

    # price -> YYYY-MM-DD string
    price_reset["trade_date"] = pd.to_datetime(price_reset["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # embeddings 必须有 trade_date
    if "trade_date" not in emb_df.columns:
        raise ValueError(f"embeddings 缺少 trade_date 列。现有列：{list(emb_df.columns)[:50]}")

    emb = emb_df.copy()

    # embeddings -> YYYY-MM-DD string
    emb["trade_date"] = pd.to_datetime(emb["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 丢掉无法解析的日期行（否则 merge 会带来空键）
    price_reset = price_reset.dropna(subset=["trade_date"])
    emb = emb.dropna(subset=["trade_date"])

    merged = price_reset.merge(emb, on="trade_date", how="inner")
    merged = merged.set_index("trade_date")
    merged.index.name = "trade_date"
    return merged


def _add_market_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    调用 compute_states_from_df 增加市场状态列。
    在进入 HMM 前强制保证 open/high/low/close 存在
    """
    print("计算市场状态...")

    df = df.copy()

    # ---- 1) 统一 OHLC 列名（解决 close_x/close_y 等）
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

    # 写回标准列名
    if close_col != "close":
        df["close"] = df[close_col]
    if open_col != "open":
        df["open"] = df[open_col]
    if high_col != "high":
        df["high"] = df[high_col]
    if low_col != "low":
        df["low"] = df[low_col]

    # 确保数值类型（避免字符串/对象）
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 2) 准备 date 列（compute_states_from_df 需要）
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})

    # date 也转成 datetime（有些情况下 index 是字符串）
    df_reset["date"] = pd.to_datetime(df_reset["date"], errors="coerce")

    # 丢掉关键字段为空的行（否则 hmm 特征会出 NaN）
    df_reset = df_reset.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)

    # ---- 3) 打印一下，确保 close 已存在
    print("[DEBUG] HMM input columns:", df_reset.columns.tolist())
    print(df_reset[["date", "open", "high", "low", "close"]].head())

    # ---- 4) 调用状态识别
    df_with_states = compute_states_from_df(df_reset)

    # ---- 5) 设回索引
    df_with_states = df_with_states.set_index("date")
    df_with_states.index.name = "trade_date"
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
            # 实在没有就补 0（MetaModel 会检查 NaN，但 0 不会 NaN）
            df["vol"] = 0.0

    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)
    return df


def main():
    # ========== 1. 加载行情数据 ==========
    fetcher = DataFetcher()

    # 行情文件名（相对于 data/ 目录） - 可以改成任意你想回测的股票数据文件
    price_filename = "601788_SH.csv"
    price_df = _load_price_data(fetcher, price_filename)

    # ========= 2. 加载 embeddings（qt_* 等特征）并合并 ==========
    embeddings_path = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\embeddings_ht_qt.csv"
    emb_df = _load_embeddings(embeddings_path)

    print("合并行情数据与 embeddings 特征...")
    df = _merge_price_and_embeddings(price_df, emb_df)

    # ========== 3. 添加市场状态 ==========
    df_with_states = _add_market_states(df)
    df_with_states = _standardize_volume(df_with_states)

    df_with_states = df_with_states.sort_index()
    df_with_states["t_index"] = np.arange(len(df_with_states), dtype=int)
    # ========== 4. 定义策略组合 ==========
    strategies = {
        # 基础策略
        "MA": MovingAverageStrategy(short_window=5, long_window=20),

        # 状态驱动策略（不使用预测）
        "StateDriven_NoPred": StateDrivenStrategy(use_prediction=False),

        # 状态驱动策略（使用预测）
        "StateDriven_WithPred": StateDrivenStrategy(use_prediction=True),

        # 元模型策略
        "MetaModel": MetaModelStrategy(
        lstm_path='utils/models/lstm_model.h5',
        hmm_path='utils/models/hmm_model.pkl',
        transformer_path='utils/models/transformer_model_state_dict.pt',
        lookback=60,
        selection_mode='three_models',
        pred_kind="logret",   # ✅ 关键
        ),

        # RAG 检索增强策略（FAISS 常驻内存 + 每天检索 topk 融合决策）
        "RAGEnhanced": RAGEnhancedStrategy(
            memory_dir=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\new\601788",
            topk=30,
            min_gap=10,
            alpha=0.5,
            buy_th=0.55,
            sell_th=0.45,
            # 风控参数（先算出来；若 Backtester 暂不支持 SL/TP，可先忽略）
            sl_mult=2.0,
            tp_mult=3.0,
            sl_floor=0.02,
            tp_floor=0.03,
            sl_cap=0.12,
            tp_cap=0.20,
        ),
    }

    # ========== 5. 运行回测 ==========
    print("开始回测...")
    backtester = Backtester(
        strategies=strategies,
        data=df_with_states,  # 注意：必须包含 qt_*（已通过 merge 保证）
        initial_capital=100000.0,
    )
    backtester.run_backtest()


if __name__ == "__main__":
    main()
