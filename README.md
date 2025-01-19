# 股票交易回测项目

本项目是一个基于多种策略的股票交易回测框架，支持多种模型（如Transformer、LSTM、HMM）生成交易信号，并进行回测分析。

## 功能简介

- **多模型支持**：包括Transformer、LSTM、HMM模型的交易信号生成。
- **策略回测**：支持多种交易策略的回测并比较收益率。
- **可视化**：绘制回测结果的收益曲线。
- **GPU加速**：支持TensorFlow和PyTorch的GPU加速。

## 文件结构

```
Stock_Trade/
├── data/               # 存放股票历史数据
├── utils/              # 工具文件夹（模型保存、加载等）
├── strategies/         # 策略实现代码
│   ├── base_strategy.py
│   └── meta_model_strategy.py
├── backtest/           # 回测模块
│   └── backtester.py
├── main.py             # 项目主入口
└── README.md           # 项目说明文件
```

## 安装依赖

### 使用Conda生成依赖文件

```bash
conda env export --no-builds > requirements.txt
```

### 安装依赖

使用以下命令安装项目所需的Python依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

1. 克隆项目：

```bash
git clone https://github.com/your-username/Stock_Trade.git
```

2. 进入项目目录：

```bash
cd Stock_Trade
```

3. 运行主程序：

```bash
python main.py
```

## 模型说明

### Transformer 模型
基于PyTorch实现的简单Transformer模型，用于捕捉股票数据中的时间序列特征。

### LSTM 模型
使用TensorFlow实现的LSTM模型，适合处理时间序列数据。

### HMM 模型
基于Hidden Markov Model（HMM）的模型，用于建模股票价格的隐含状态转移。

## 回测策略

### MetaModel 策略
- 动态选择LSTM、Transformer或HMM模型，根据最近的市场波动和收益率特征选择最优模型。
- 使用生成的信号进行回测分析。

## 注意事项

- 请确保您的数据文件位于`data/`目录下，格式包括`open`、`high`、`low`、`close`、`vol`等列。
- 若需要启用GPU，请确保您的环境已正确配置CUDA。

## 示例结果

回测运行后将生成以下可视化结果：

1. 策略收益曲线对比。
2. 每个策略的最终收益率统计。

## Appendix

https://github.com/wondertrader/wondertrader
https://github.com/UFund-Me/Qbot
https://github.com/yutiansut/QUANTAXIS
https://github.com/hugo2046/QuantsPlaybook
https://github.com/thuquant/awesome-quant?tab=readme-ov-file#%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E5%B9%B3%E5%8F%B0

## 许可证

本项目基于MIT许可证发布。

## 20250119计划变动

- 修改：在strategies中bollinger和ma等等需要转到交易策略中，跟预测策略随机组合
- nlp_sentiment: 加上每个预测策略，权重给股民，大V，政府。

