# base_strategy.py
import pandas as pd

class BaseStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        每个策略都应实现本方法：
        输入: (历史)DataFrame
        输出: 一个与 df 行数相同的信号序列(可以是-1到1浮动，或0/1，等)
        对于A股做多场景，可以仅使用 [0, 1] 代表空仓/满仓，或 [0, 0.5, 1] 代表持仓比例。
        """
        raise NotImplementedError
