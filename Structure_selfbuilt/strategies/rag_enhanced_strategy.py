import numpy as np
import pandas as pd
from utils.rag_memory import RAGMemory


class RAGEnhancedStrategy:
    """
    输出：每个交易日的“目标仓位”Series（index=df.index）
    Backtester 会用 t-1 的仓位去乘 t 的真实收益。

    设计目标：
    - memory_dir: 板块 RAG 经验库（FAISS + meta）
    - df: 目标个股自身的 encoder 输出（至少包含 qt_*、y_hat、t_index，可选 ts_code）
    - 每天用当前 qt 检索板块历史经验，再和当前 y_hat 融合，得到 position
    """

    def __init__(
        self,
        memory_dir: str,
        topk: int = 30,
        min_gap: int = 10,
        alpha: float = 0.5,
        buy_th: float = 0.55,
        sell_th: float = 0.45,
        sl_mult: float = 2.0,
        tp_mult: float = 3.0,
        sl_floor: float = 0.02,
        tp_floor: float = 0.03,
        sl_cap: float = 0.12,
        tp_cap: float = 0.20,
        strict_past_only: bool = True,   # 强烈建议：只允许检索“过去”的经验
        exclude_self: bool = True,       # 是否排除目标个股自身历史
    ):
        self.mem = RAGMemory(
            memory_dir=memory_dir,
            topk=topk,
            min_gap=min_gap,
            alpha=alpha,
            buy_th=buy_th,
            sell_th=sell_th,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            sl_floor=sl_floor,
            tp_floor=tp_floor,
            sl_cap=sl_cap,
            tp_cap=tp_cap,
        )

        self.strict_past_only = strict_past_only
        self.exclude_self = exclude_self

        # debug：保留最后一次决策
        self.last_decision = None
        self.last_topk = None

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        qt_cols = [c for c in df.columns if c.startswith("qt_")]
        if len(qt_cols) == 0:
            raise ValueError("RAGEnhancedStrategy: df 中找不到 qt_* 列（encoder 输出缺失）。")

        # t_index 是时间过滤的关键，没有就补
        if "t_index" not in df.columns:
            df["t_index"] = np.arange(len(df), dtype=int)

        # trade_date 可选，但建议有
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

        # y_hat 没有的话，可以兜底成 0.5
        if "y_hat" not in df.columns:
            df["y_hat"] = 0.5

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        返回目标仓位 series（0..1）
        Backtester 用 t-1 仓位作用到 t 收益，所以这里不做 shift。
        """
        df = self._ensure_required_columns(df)

        positions = pd.Series(index=df.index, dtype=float)

        # 尝试识别目标股票代码
        target_ts_code = None
        if "ts_code" in df.columns and df["ts_code"].notna().any():
            target_ts_code = str(df["ts_code"].dropna().iloc[0])

        for i in range(len(df)):
            # 第一天无法形成有效历史，先空仓
            if i == 0:
                positions.iloc[i] = 0.0
                continue

            # 是否排除自身股票
            exclude_ts_code = target_ts_code if self.exclude_self else None

            # 查询 RAG + Transformer 融合决策
            dec = self.mem.query_one(
                df=df,
                q_idx=i,
                exclude_ts_code=exclude_ts_code,
                strict_past_only=self.strict_past_only,
            )

            self.last_decision = dec

            # 如果 query_one 返回带 topk 详情，可以顺手记下来
            if hasattr(dec, "topk_df"):
                self.last_topk = dec.topk_df

            positions.iloc[i] = float(dec.position)

        return positions

