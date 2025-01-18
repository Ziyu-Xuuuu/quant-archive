# bollinger_strategy.py
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


class BollingerStrategy(BaseStrategy):
    def __init__(self, period=20, num_std=2):
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        布林带策略:
        - 计算中轨(均线)、上轨、中轨，下轨
        - 当价格 < 下轨 -> 满仓
        - 当价格 > 上轨 -> 空仓
        - 其余 -> 0.5 仓位(示例)
        """
        df = df.copy()
        df['ma'] = df['close'].rolling(self.period).mean()
        df['std'] = df['close'].rolling(self.period).std()
        df['upper'] = df['ma'] + self.num_std * df['std']
        df['lower'] = df['ma'] - self.num_std * df['std']

        # 初期NAN处理
        df[['ma', 'std', 'upper', 'lower']] = df[['ma', 'std', 'upper', 'lower']].fillna(method='bfill')

        signal = pd.Series(0.5, index=df.index)  # 默认0.5仓
        signal[df['close'] < df['lower']] = 1.0  # 满仓
        signal[df['close'] > df['upper']] = 0.0  # 空仓
        return signal
