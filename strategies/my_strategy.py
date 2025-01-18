# ma_strategy.py
import pandas as pd

class MovingAverageStrategy:
    def __init__(self, short_window=5, long_window=20):
        """
        :param short_window: 短期均线窗口
        :param long_window:  长期均线窗口
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        """
        基于双均线金叉死叉生成交易信号:
        1 -> 买入
        -1 -> 卖出
        0 -> 无操作
        """
        df = df.copy()
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()

        df['signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
        df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1

        return df['signal']

