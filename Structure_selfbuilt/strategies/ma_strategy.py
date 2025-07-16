# ma_strategy.py
import pandas as pd
from strategies.base_strategy import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        双均线策略, 信号范围[0,1]:
        - 如果短均线 > 长均线, 信号=1 (满仓)
        - 否则信号=0 (空仓)
        """
        df = df.copy()
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()

        # 注意rolling会产生NaN，先填充或丢弃初期数据
        df[['ma_short','ma_long']] = df[['ma_short','ma_long']].fillna(method='bfill')

        signal = (df['ma_short'] > df['ma_long']).astype(float)
        return signal
