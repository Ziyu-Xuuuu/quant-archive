# strategies/hmm_meta_hybrid_strategy.py
import pandas as pd
from strategies.base_strategy import BaseStrategy

class HMMMetaHybridStrategy(BaseStrategy):
    """
    使用:
      - HMM 状态: state / trend_regime / vol_regime
      - MetaModel 预测: meta_predicted_close
    来生成交易信号的“组合策略”
    """

    def __init__(self, up_threshold: float = 0.02, down_threshold: float = -0.02):
        """
        up_threshold: 预测涨幅超过这个阈值才认为是有效做多信号
        down_threshold: 预测跌幅低于这个阈值才认为是有效做空/减仓信号
        """
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()

        # 1. 检查数据是否有所需列
        required_cols = ['close', 'state', 'trend_regime']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"HMMMetaHybridStrategy 缺少必需列: {missing} "
                             f"请确认已调用 compute_states_from_df 生成状态。")

        # 如果没有预测列，则用空仓信号
        if 'meta_predicted_close' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # 根据预测值和状态来生成信号
        price = df['close']
        pred = df['meta_predicted_close']
        expected_ret = (pred / price) - 1.0

        trend = df['trend_regime'].astype(str)
        path = df['state'].astype(str)

        signal = pd.Series(0.0, index=df.index, dtype=float)

        # === 多头环境：T1 + S1 / S2 ===
        long_env = (trend == "T1") & (path.isin(["S1", "S2"]))
        signal[long_env & (expected_ret > self.up_threshold)] = 1.0

        # === 空头环境：T3 + S3 / S4 ===
        short_env = (trend == "T3") & (path.isin(["S3", "S4"]))
        signal[short_env & (expected_ret < self.down_threshold)] = 0.0

        return signal.fillna(0.0)
