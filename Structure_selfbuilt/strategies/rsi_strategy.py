# rsi_strategy.py
import pandas as pd
from strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, rsi_upper=70, rsi_lower=30):
        self.period = period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        RSI策略示例:
        - 当RSI < rsi_lower 时，认为超卖 -> 买入(满仓=1)
        - 当RSI > rsi_upper 时，认为超买 -> 卖出(空仓=0)
        - 其他情况 -> 部分仓位(举例: 0.5)
        """
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / (loss + 1e-9)  # 避免除零
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # 初始填充

        signal = pd.Series(0.5, index=df.index)  # 默认0.5仓
        signal[df['rsi'] < self.rsi_lower] = 1.0  # 满仓
        signal[df['rsi'] > self.rsi_upper] = 0.0  # 空仓
        return signal
