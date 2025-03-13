# Quantitative polymerization model （量化聚合模型）

本项目是一个基于多种策略的量化交易框架，支持多种模型生成交易信号，并进行回测分析。

## Ⅰ. 功能简介

- **多模型支持**：包括多种模型的交易信号生成。
- **策略回测**：支持多种交易策略的回测并比较收益率。
- **GPU加速**：支持TensorFlow和PyTorch的GPU加速。

## Ⅱ. 文件结构

``` plaintext
STOCK_QUANT/
├── backtest/           # 回测模块，用于回测模型效果
│   └── backtester.py
├── config/             # 证券接入模块，用于与真实市场接入
│   └── config.py
├── data/               # 存放股票历史数据，作为数据集使用
│   └── 股票代码.csv
├── live/               # 交易模块，根据预测决定买卖行为
│   └── trade.py
├── Python_Crawler      # 爬虫程序，用于爬取网络数据
│   ├── Micro_blog      # 爬取微博评论
│   └── Snow_Ball       # 爬取雪球评论
├── QuantAI             # 用于量化分析的智能体
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

## Ⅲ.快速开始

### 安装Git

详细步骤参考：[Git 详细安装教程（详解 Git 安装过程的每一个步骤）](https://blog.csdn.net/mukes/article/details/115693833)

## Ⅳ. 环境配置

### *. 虚拟环境软件准备与什么是虚拟环境

虚拟环境是用来为了避免与主环境相冲突而在电脑中隔离出一块全新环境来进行代码作业。需要下载anaconda软件并完成环境配置，详细请参考：[安装conda搭建python环境（保姆级教程）](https://blog.csdn.net/Q_fairy/article/details/129158178)

完成后，以下步骤请打开cmd（命令行窗口）运行

### 1. 进入项目目录

```bash
cd /your_local_address/Stock_Quant
```

### 2. 创建含有所需依赖的虚拟环境

运行命令行前，将Stock_Quant_environment.yml先下载下来放在命令行运行的文件夹里

```bash
conda env create -f Stock_Quant_environment.yml
```

### 3. 打开/关闭虚拟环境

分别使用以下命令打开/关闭虚拟环境：

```bash
conda activate Stock_Quant_environment
```

```bash
conda deactivate Stock_Quant_environment
```

### 4. 进入项目目录

```bash
git clone https://github.com/Ziyu-Xuuuu/Stock_Quant.git
```

```bash
cd Stock_Quant
```

### 5. 环境维护与更新

环境依赖的更新在Stock_Quant_environment.yml文件中进行
并通过以下代码更新环境以达到修改过后的依赖要求

```bash
conda env update -f Stock_Quant_environment.yml --name Stock_Quant_environment
```

### 6. VScode上使用git修改代码并上传

在本地的git库中对代码完成修改并保存后在VScode的终端中输入以下代码

```plaintext
git add '你修改的文件名' ##记得加上文件格式（这一步是将文件上传到缓冲区，有多少个文件上传多少个）
git commit -m '注释' ##单引号里加上本次更新的一些注释（这一步是将更新上传到本地的git库）
```

然后在VScode上使用源码管理功能进行更新同步，若源码管理显示找不到git可以看下面的常见问题解决3.

同时也可以使用该功能远程拉取等等。

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

```plaintext
直接复制粘贴报错内容给GPT，一般问题与解决策略包括但不限于：
-缺少必要前置环境，如C++ Windows；Pytorch等，按照教程官网安装即可，同时麻烦将需要前置条件的库在requirement.txt备注中说明。
-网速过慢，安装库所需时间以小时计。这可能与梯子有关，按照GPT提供的用国内镜像网址安装即可。
-环境冲突（现在经过改版的requirement.txt正常来说可以避免）。请按照GPT建议的进行环境升降级，同时麻烦将最后正确的环境型号加入到requirement.txt文件中去。
```

### 2.我的git出bug了

```plaintext
一般问题与解决策略包括但不限于：
-网速过慢，出现 "Failed to connect to github.com port 443 after 21051 ms: Could not connect to server"。这可能与梯子有关，请打开科学上网进行配置，同时需要vpn全局代理否则cmd窗口命令行仍然无法代理。
若实在走投无路，可以直接从 https://github.com/Ziyu-Xuuuu/Stock_Quant.git 下载代码到本地应急，但是将无法实时编辑。
-成功克隆到本地但是编辑后无法上传，提示未给Git配置账号。打开所用软件（Vscode, Pycharm等）终端，输入以下代码：
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
来进行全局git配置
```

### 3.VScode找不到git

参考这篇文章: [vscode找不到git，使用不了git，如何配置git](https://blog.csdn.net/weixin_38779534/article/details/113112529)

## *Appendix

目前较为完善的相关git社区：

### 1.WonderTrader

详情访问：[WonderTrader](https://github.com/wondertrader/wondertrader)

#### **WonderTrader 是什么？**

**WonderTrader** 是一个基于 **C++ 核心模块**，适用于全市场**全品种交易**的**高效率、高可用**的量化交易开发框架。

##### **主要特点**

- **面向专业机构的整体架构**
- **支持数十亿级的实盘管理规模**
- **涵盖量化交易的全流程**：  
  - 数据清洗  
  - 回测分析  
  - 实盘交易  
  - 运营调度  

**WonderTrader** 提供了从**研究、交易到运营管理**的完整一站式量化开发环境，适用于专业机构和高频交易策略开发。

##### **技术优势**

- **C++ 核心框架**：高性能、低延迟
- **基于 wtpy 框架**：支持 Python 接口，方便策略开发
- **UFT 交易引擎**（0.9 版本引入）：专为**超低延迟交易**优化，交易延迟控制在 **175 纳秒** 以内

##### **适用场景**

**WonderTrader** 适用于：

- 高频交易（HFT）
- 量化对冲
- 多资产交易（股票、期货、外汇等）

### 2.Qbot

详情访问：[Qbot](https://github.com/UFund-Me/Qbot)

#### Qbot - AI智能量化投研平台

**Qbot** 是一个 AI 驱动的自动化量化投资研究平台，旨在挖掘 AI 技术在量化投资中的潜力，并赋能投资者。Qbot 支持多种机器学习建模范式，包括：

- 监督学习（Supervised Learning）
- 市场动态建模（Market Dynamics Modeling）
- 强化学习（Reinforcement Learning, RL）

##### 平台特点

- **智能分析**：结合 AI 算法分析股票等金融资产，提供量化投资建议。
- **数据驱动**：基于财务数据、市场动态及历史趋势进行建模，提高投资决策的科学性。
- **自动化处理**：支持 PEG、ROE、EPS 等财务指标分析，辅助投资者评估资产价值。

##### 功能示例

Qbot 可对个股进行财务指标检测，例如：

- EPS（每股收益）是否呈现增长趋势
- PEG（市盈增长比率）是否低于 0
- ROE（净资产收益率）是否高于一定阈值

### 3.QUANTAXIS 2.0.0

详情访问：[QUANTAXIS 2.0.0](https://github.com/yutiansut/QUANTAXIS)

#### QUANTAXIS 2.0.0

##### 什么是 QUANTAXIS？

QUANTAXIS 是一个 **量化金融框架（Quantitative Financial Framework）**，提供多市场数据支持、交易管理、账户体系等功能，帮助量化研究员和投资者进行数据驱动的投资研究。

##### 核心模块

QUANTAXIS 主要由以下几个核心模块组成：

###### 1.QASU / QAFetch

- 支持 **多市场数据存储、自动运维、数据获取**
- 兼容 **MongoDB / ClickHouse** 数据库，提供高效数据存储方案

###### 2.QAUtil

- 支持 **交易时间管理、交易日历**
- 提供 **时间前向后推算、市场识别、DataFrame 数据转换** 等工具

###### 3.QIFI / QAMarket —— 多市场、多语言、统一账户体系

- **qifiaccount**：标准账户体系，支持 Rust/Cpp 版本，保证跨平台一致性
- **qifimanager**：多账户管理体系，支持 **多个语言的账户统一管理**
- **qaposition**：单标的仓位管理模块，支持 **精准多空控制**（套利/CTA/股票等场景）
- **marketpreset**：市场预制基类，便于 **查询期货、股票、虚拟货币、品种 tick 数据、保证金、手续费** 等
