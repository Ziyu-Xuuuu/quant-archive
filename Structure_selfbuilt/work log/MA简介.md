## 均线策略 (Moving Average, MA，预测策略)

### **基础原理**

均线是某段时间内价格的平均值，用于平滑价格波动，识别趋势方向。它分为**短期均线**和**长期均线**，结合两者的交叉情况来发出买卖信号。一般来说，长期均线代表的是股票的长期走向，而短期均线代表的是最近的波动。具体分为两种情况：

1. 如果是短期均线上穿长期均线，说明股票最近的走势比较好，可以买入
2. 如果是短期均线下穿长期均线，说明股票最近的走势比较好，可以卖出

### **适用场景**

- **趋势市场**：价格有明显上涨或下跌趋势。（如果是长期均线下跌趋势的话比较难说，因为只能保证跌得没那么多，这是可以采用的是hold或者stay）
- **中长期投资**：平滑价格波动，过滤短期噪声。
- 不适合震荡市场，容易产生频繁的错误信号。

### **特点**

- 简单易用，广泛适用于多种市场。
- 滞后性：均线的变化滞后于价格变化。
- 超参数的设计比较重要，长期和短期的界定需要经验。

### **公式**

$$
MA_t = \frac{P_{t-n+1} + P_{t-n+2} + \dots + P_t}{n}
$$

- (MA_t\)：第 \(t\) 天的移动平均值。
- \(P_t\)：第 \(t\) 天的收盘价。（开盘也行）
- \(n\)：时间窗口。

### **交易规则**

1. **金叉（买入）**：短期均线上穿长期均线。
2. **死叉（卖出）**：短期均线下穿长期均线。

### **实现代码**


```
import pandas as pd

class MAStrategy:
    def __init__(self, short_window=10, long_window=50):
        """
        初始化均线策略
        :param short_window: 短期均线窗口
        :param long_window: 长期均线窗口
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        """
        生成 MA 策略交易信号
        :param df: 包含历史数据的 DataFrame，需有 'close' 列
        :return: 包含信号的 DataFrame
        """
        # 计算短期和长期均线
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
  
        # 生成交易信号
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # 买入信号
        df.loc[df['short_ma'] <= df['long_ma'], 'signal'] = -1  # 卖出信号
  
        return df
```

import pandas as pd

class MAStrategy:
    def __init__(self, short_window=10, long_window=50):
        """
        初始化均线策略
        :param short_window: 短期均线窗口
        :param long_window: 长期均线窗口
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        """
        生成 MA 策略交易信号
        :param df: 包含历史数据的 DataFrame，需有 'close' 列
        :return: 包含信号的 DataFrame
        """
        # 计算短期和长期均线
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
  
        # 生成交易信号
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # 买入信号
        df.loc[df['short_ma'] <= df['long_ma'], 'signal'] = -1  # 卖出信号
  
        return df


### 后期改进

1. 01信号可以转化为连续信号
2. 优化长期均线跌的情况
