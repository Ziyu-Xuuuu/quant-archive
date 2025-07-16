# macd_strategy.py
import pandas as pd
from strategies.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.fast = fastperiod
        self.slow = slowperiod
        self.signal = signalperiod

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        MACD策略示例，信号范围[0,1]:
        - DIFF > DEA (即MACD>0) -> 1
        - 否则 -> 0
        """
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow).mean()
        df['diff'] = df['ema_fast'] - df['ema_slow']
        df['dea'] = df['diff'].ewm(span=self.signal).mean()
        df['macd'] = 2 * (df['diff'] - df['dea'])  # 传统MACD柱线

        # 这里仅示例：DIFF>DEA时满仓，否则空仓
        signal = (df['diff'] > df['dea']).astype(float)
        return signal
