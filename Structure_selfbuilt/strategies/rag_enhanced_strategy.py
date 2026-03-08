import numpy as np
import pandas as pd
from utils.rag_memory import RAGMemory

class RAGEnhancedStrategy:
    """
    输出：每个交易日的“目标仓位”Series（index=df.index）
    Backtester 会用 t-1 的仓位去乘 t 的真实收益。
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
        strict_past_only: bool = True,   # ✅ 强烈建议：只允许检索过去，杜绝泄漏
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

        # debug：保留最后一次决策
        self.last_decision = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        返回 position series（0..1）
        """
        df = df.copy()

        # 必要列检查
        qt_cols = [c for c in df.columns if c.startswith("qt_")]
        if len(qt_cols) == 0:
            raise ValueError("RAGEnhancedStrategy: df 中找不到 qt_* 列（encoder 输出缺失）。")
        if "t_index" not in df.columns:
            # 给一个兜底 t_index（建议你在 pipeline 显式提供）
            df["t_index"] = np.arange(len(df), dtype=int)

        positions = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            # 第一天先空仓
            if i == 0:
                positions.iloc[i] = 0.0
                continue

            # 严格只用过去样本（强烈推荐）
            dec = self.mem.query_one(df, q_idx=i)
            self.last_decision = dec

            # dec.position 就是 0..1
            positions.iloc[i] = float(dec.position)

        # Backtester 用 t-1 仓位作用于 t 的收益，因此这里返回“当日决定的仓位”即可
        # Backtester 已经用 i-1，所以不用 shift。
        return positions
