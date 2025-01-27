# Quantitative polymerization model （量化聚合模型）

本项目是一个基于多种策略的量化交易框架，支持多种模型生成交易信号，并进行回测分析。

## Ⅰ. 功能简介

- **多模型支持**：包括多种模型的交易信号生成。
- **策略回测**：支持多种交易策略的回测并比较收益率。
- **GPU加速**：支持TensorFlow和PyTorch的GPU加速。

## Ⅱ. 文件结构

```
STOCK_QUANT/
├── backtest/           # 回测模块，用于回测模型效果
│   └── backtester.py
├── config/             # 证券接入模块，用于与真实市场接入
│   └── config.py
├── data/               # 存放股票历史数据，作为数据集使用
│   └── 股票代码.csv
├── live/               # 交易模块，根据预测决定买卖行为
│   └── trade.py
├── Stock/              # 暂无
├── strategies/         # 预测模块，包含多种常用模型
│   └── xxmodel_strategy.py
├── utils/              # 预训练文件夹（模型保存、预训练等）
│   ├── models          # 模型参数
│   └── train           # 模型预训练
├── work log            # 工作日志
├── main.py             # 项目主入口
├── README.md           # 项目说明文件
└── requirement.txt     # 环境配置参数
```

## Ⅲ. 环境配置（以stock_quant_env这个虚拟环境为例）

### *. 虚拟环境软件准备与什么是虚拟环境

虚拟环境是用来为了避免与主环境相冲突而在电脑中隔离出一块全新环境来进行代码作业。需要下载anaconda软件。

以下步骤请打开cmd（命令行窗口）运行

### 1. 使用Conda生成新隔离环境

```bash
conda create --name stock_quant_env python=3.10.16
```

*验证是否创建了虚拟环境：打开anaconda文件夹，找到envs文件夹，看其中是否有你所创建的虚拟环境名的文件夹。
e.g, "C:\Users\24746\anaconda3\envs\stock_quant_env"

### 2. 激活虚拟环境

```bash
conda activate stock_quant_env
```

### 3. 安装依赖

分别使用以下命令安装项目所需的Python依赖和C依赖，具体见requirements. txt：

```bash
pip install       # python相关依赖
conda install     # C相关依赖
```

### 4. 环境配置及维护方法的更新

现已将requirement.txt更新整理为Stock_Quant_environment.yml。可直接通过以下代码创建符合依赖要求的conda environment

```
conda env create -f Stock_Quant_environment.yml
```

或通过以下代码更新环境以达到修改过后的依赖要求

```
conda env update -f Stock_Quant_environment.yml --name Stock_Quant_environment
```

建议此后环境依赖的更新都在此.yml文件中进行
（注：此方法仍存在bug，在调试成功后会另行说明）

## Ⅳ.快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Ziyu-Xuuuu/Stock_Quant.git
```

### 2. 进入项目目录

```bash
cd Stock_Quant
```

## Ⅴ. 模型介绍

### 1. 预测模型

- Transformer 模型
- LSTM 模型
- HMM 模型

### 2. 交易策略

- bollinger 策略
- 双均线策略
- MACD策略

## Ⅵ. 回测策略

### MetaModel 策略

- 动态选择LSTM、Transformer或HMM模型，根据最近的市场波动和收益率特征选择最优模型。
- 使用生成的信号进行回测分析。

## Ⅷ. 示例结果

回测运行后将生成以下可视化结果：

1. 策略收益曲线对比。
2. 每个策略的最终收益率统计。

## *常见问题（请大家将各自遇到的配置问题加入以便其他协作者使用）

### 1.环境配置时我的cmd出bug了

```
直接复制粘贴报错内容给GPT，一般问题与解决策略包括但不限于：
-缺少必要前置环境，如C++ Windows；Pytorch等，按照教程官网安装即可，同时麻烦将需要前置条件的库在requirement.txt备注中说明。
-网速过慢，安装库所需时间以小时计。这可能与梯子有关，按照GPT提供的用国内镜像网址安装即可。
-环境冲突（现在经过改版的requirement.txt正常来说可以避免）。请按照GPT建议的进行环境升降级，同时麻烦将最后正确的环境型号加入到requirement.txt文件中去。
```

### 2.我的git出bug了

```
一般问题与解决策略包括但不限于：
-网速过慢，出现 "Failed to connect to github.com port 443 after 21051 ms: Could not connect to server"。这可能与梯子有关，请打开科学上网进行配置，同时需要vpn全局代理否则cmd窗口命令行仍然无法代理。
若实在走投无路，可以直接从 https://github.com/Ziyu-Xuuuu/Stock_Quant.git 下载代码到本地应急，但是将无法实时编辑。
-成功克隆到本地但是编辑后无法上传，提示未给Git配置账号。打开所用软件（Vscode, Pycharm等）终端，输入以下代码：
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
来进行全局git配置
```

## *Appendix

https://github.com/wondertrader/wondertrader
https://github.com/UFund-Me/Qbot
https://github.com/yutiansut/QUANTAXIS
https://github.com/hugo2046/QuantsPlaybook
https://github.com/thuquant/awesome-quant?tab=readme-ov-file#%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E5%B9%B3%E5%8F%B0
