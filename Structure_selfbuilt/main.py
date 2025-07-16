import os
from config.config import BROKER_CONFIG
from utils.data_fetcher import DataFetcher

# 交易策略
from strategies.ma_strategy import MovingAverageStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_strategy import BollingerStrategy

# 预测模型
from strategies.meta_model_strategy import MetaModelStrategy
from strategies.nlp_sentiment_strategy import NLPSentimentStrategy

from backtest.backtester import Backtester
from live.trader import LiveTrader

def main():
    # ========== 1. 获取或加载历史数据 ==========
    fetcher = DataFetcher()
    filename = "601788_SH.csv"
    file_path = r"C:\Users\user\Documents\GitHub\trader\Stock_Trade\data\601788_SH.csv"

    if not os.path.exists(file_path):
        print(f"未检测到 {filename}，开始下载历史数据...")
        df = fetcher.get_historical_data(ts_code="000001.SZ", start_date="20200101", end_date="20211231")
        fetcher.save_data_to_csv(df, filename)
    else:
        print(f"检测到 {filename}，直接读取本地数据...")
        df = fetcher.read_data_from_csv(file_path)

    # ========== 2. 多策略回测比较 ==========
    # 2. 多策略回测比较
    print("开始回测 ...")

    # 在这里定义你想要对比的多种 selection_mode
    selection_modes = [
        'three_models',
        'lstm_hmm',
        'lstm_transformer',
        'hmm_transformer'
    ]

    for mode in selection_modes:
        print(f"=== 当前 selection_mode: {mode} ===")

        # 每个模式都单独构建一次策略字典
        strategies = {
            # 交易策略（会实际产生资金曲线）
            "MA": MovingAverageStrategy(short_window=5, long_window=20),
            "MACD": MACDStrategy(),
            "RSI": RSIStrategy(period=14, rsi_upper=70, rsi_lower=30),
            "Bollinger": BollingerStrategy(period=20, num_std=2),

            # 将 MetaModel 的 selection_mode 改为当前循环的 mode
            "MetaModel": MetaModelStrategy(
                lstm_path='utils/models/lstm_model.h5',
                hmm_path='utils/models/hmm_model.pkl',
                transformer_path='utils/models/transformer_model_state_dict.pt',
                lookback=60,
                selection_mode=mode  # <-- 关键：传入不同的组合模式
            ),
            "NLP": NLPSentimentStrategy()
        }

        # 每次都生成一个新的 Backtester 进行回测
        backtester = Backtester(strategies=strategies, data=df, initial_capital=100000.0)
        backtester.run_backtest()

    # ========== 3. 实盘(模拟)交易 ==========
    print("开始模拟实盘交易(示例) ...")
    # 只对 MetaModel 做模拟交易：你也可以换成别的
    meta_strategy = MetaModelStrategy(
        lstm_path='utils/models/lstm_model.h5',
        hmm_path='utils/models/hmm_model.pkl',
        transformer_path='utils/models/transformer_model_state_dict.pt',
        lookback=60
    )
    live_trader = LiveTrader(strategy=meta_strategy, broker_config=BROKER_CONFIG)
    live_trader.start_trading()

if __name__ == "__main__":
    main()
