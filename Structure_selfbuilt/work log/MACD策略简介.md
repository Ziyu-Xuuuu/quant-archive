# MACD 策略 (Moving Average Convergence Divergence)

## **基础原理**

MACD 通过两条指数移动平均线（EMA）分析短期与长期动量的差值，捕捉趋势反转信号。类似于MA策略，但是加上了EMA，更符合短中期波动。

## **适用场景**

- **趋势市场**：捕捉趋势方向变化。
- 适合中长期趋势交易。

## **特点**

- 对趋势反转敏感，适合捕捉动量信号。
- 容易在震荡市场中产生误导信号。

## **公式**

* **快线 (DIF)**：

  $$
  DIF = \text{短期EMA} - \text{长期EMA}
  $$
* **慢线 (DEA)（用于平滑DIF）**：

  $$
  DEA = DIF \text{ 的 EMA}
  $$
* **柱状图**：

  $$
  柱状图 = DIF - DEA
  $$


## **交易规则**

1. **买入信号**：DIF 上穿 DEA，柱状图变正。
2. **卖出信号**：DIF 下穿 DEA，柱状图变负。

## **实现代码**

```python
class MACDStrategy:
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        """
        初始化 MACD 策略
        :param short_window: 短期 EMA 窗口
        :param long_window: 长期 EMA 窗口
        :param signal_window: 信号线窗口
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def generate_signals(self, df):
        """
        生成 MACD 策略交易信号
        :param df: 包含历史数据的 DataFrame，需有 'close' 列
        :return: 包含信号的 DataFrame
        """
        # 计算 EMA
        df['ema_short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
  
        # 计算 DIF 和 DEA
        df['dif'] = df['ema_short'] - df['ema_long']
        df['dea'] = df['dif'].ewm(span=self.signal_window, adjust=False).mean()
  
        # 计算柱状图
        df['macd'] = 2 * (df['dif'] - df['dea'])
  
        # 生成信号
        df['signal'] = 0
        df.loc[df['dif'] > df['dea'], 'signal'] = 1  # 买入信号
        df.loc[df['dif'] <= df['dea'], 'signal'] = -1  # 卖出信号
  
        return df
```
