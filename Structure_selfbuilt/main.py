# main.py
import os
import pandas as pd
from config.config import BROKER_CONFIG
from utils.data_fetcher import DataFetcher

# 导入状态识别
from regime_hmm import compute_states_from_df

# 导入策略
from strategies.ma_strategy import MovingAverageStrategy
from strategies.state_driven_strategy import StateDrivenStrategy
from strategies.meta_model_strategy import MetaModelStrategy

from backtest.backtester import Backtester

def main():
    # ========== 1. 加载数据 ==========
    fetcher = DataFetcher()
    # 注意：read_data_from_csv 会自动在路径前加 'data/'
    # 所以只需传入文件名
    filename = "601788_SH.csv"
    df = fetcher.read_data_from_csv(filename)

    # 或者直接使用 pandas 读取完整路径
    # df = pd.read_csv(r"data/601788_SH.csv", index_col=0, parse_dates=True)

    # ========== 2. 添加市场状态（可选）==========
    print("计算市场状态...")
    df_reset = df.reset_index()
    # 确保第一列重命名为 'date'
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})

    # 调用状态识别函数
    df_with_states = compute_states_from_df(df_reset)

    # 将 date 设回索引，保持原有索引名
    df_with_states = df_with_states.set_index("date")
    df_with_states.index.name = "trade_date"  # 恢复原索引名

    # ========== 3. 定义策略组合 ==========
    strategies = {
        # 基础策略
        "MA": MovingAverageStrategy(short_window=5, long_window=20),

        # 状态驱动策略（不使用预测）
        "StateDriven_NoPred": StateDrivenStrategy(use_prediction=False),

        # 状态驱动策略（使用预测）
        "StateDriven_WithPred": StateDrivenStrategy(use_prediction=True),

        # 元模型策略
        "MetaModel": MetaModelStrategy(
            lstm_path='utils/models/lstm_model.h5',
            hmm_path='utils/models/hmm_model.pkl',
            transformer_path='utils/models/transformer_model_state_dict.pt',
            lookback=60,
            selection_mode='three_models'
        )
    }

    # ========== 4. 运行回测 ==========
    print("开始回测...")
    backtester = Backtester(
        strategies=strategies,
        data=df_with_states,  # 使用带状态的数据
        initial_capital=100000.0
    )
    backtester.run_backtest()

if __name__ == "__main__":
    main()