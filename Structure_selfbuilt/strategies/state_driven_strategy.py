# strategies/state_driven_strategy.py
import pandas as pd
from strategies.base_strategy import BaseStrategy

class StateDrivenStrategy(BaseStrategy):
    """
    根据市场状态动态调整交易逻辑的策略
    """
    def __init__(self, use_prediction=True):
        self.use_prediction = use_prediction

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        signal = pd.Series(0.0, index=df.index)

        # 选择使用预测价格还是真实价格
        price_col = 'meta_predicted_close' if (
            self.use_prediction and 'meta_predicted_close' in df.columns
        ) else 'close'

        # 计算技术指标
        df['ma_fast'] = df[price_col].rolling(5).mean()
        df['ma_slow'] = df[price_col].rolling(20).mean()
        df['rsi'] = self._compute_rsi(df[price_col], 14)

        # 如果有状态信息，使用状态驱动逻辑
        if 'state' in df.columns:
            for i in df.index:
                state = df.loc[i, 'state']

                if state == 'S1':  # 趋势启动期
                    # 金叉且 RSI 未超买 -> 满仓
                    if df.loc[i, 'ma_fast'] > df.loc[i, 'ma_slow'] and df.loc[i, 'rsi'] < 70:
                        signal[i] = 1.0

                elif state == 'S2':  # 趋势延续期
                    # 保持趋势方向
                    if df.loc[i, 'ma_fast'] > df.loc[i, 'ma_slow']:
                        signal[i] = 1.0

                elif state == 'S3':  # 趋势衰竭期
                    # 减仓，但不完全退出
                    if df.loc[i, 'ma_fast'] > df.loc[i, 'ma_slow']:
                        signal[i] = 0.5

                elif state == 'S4':  # 高波动震荡
                    # 空仓观望
                    signal[i] = 0.0

                elif state == 'S5':  # 恢复期
                    # RSI 超卖时轻仓试探
                    if df.loc[i, 'rsi'] < 30:
                        signal[i] = 0.3
        else:
            # 无状态信息时使用简单均线策略
            signal[df['ma_fast'] > df['ma_slow']] = 1.0

        return signal.fillna(0.0)

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi