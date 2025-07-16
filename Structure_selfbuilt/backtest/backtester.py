import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# 预测模型
from strategies.meta_model_strategy import MetaModelStrategy
from strategies.nlp_sentiment_strategy import NLPSentimentStrategy

class Backtester:
    def __init__(self, strategies: dict, data: pd.DataFrame, initial_capital=100000.0):
        """
        strategies: {策略名: 策略对象}，其中可能包含 'MetaModel' / 'NLP' 等预测模型，以及 MA/MACD/RSI/Bollinger 等交易策略
        data: 原始行情 DataFrame，含 [open, high, low, close, vol] 等
        initial_capital: 初始资金
        """
        self.strategies = strategies
        self.data = data.copy()  # 不直接改原始data
        self.initial_capital = initial_capital
        self.equity_curves = {}
        self.meta_model_used = False  # 用于判断是否执行过MetaModel
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_backtest(self):
        """
        流程:
        1) 如果有 MetaModel 策略 => 先用它对 self.data 生成 'meta_predicted_close'
        2) 其他策略基于 'meta_predicted_close' 生成买卖信号 => 计算资金曲线
        """
        # 1) 若有 MetaModel => 先执行一次
        meta_strategy_key = None
        for k, v in self.strategies.items():
            if isinstance(v, MetaModelStrategy):
                meta_strategy_key = k
                break

        if meta_strategy_key is not None:
            logging.info("先执行 MetaModel 生成预测收盘价 ...")
            meta_output = self.strategies[meta_strategy_key].generate_signals(self.data)
            # 将预测值存储到 self.data
            self.data['meta_predicted_close'] = meta_output
            self.meta_model_used = True
            # 如果想可视化K线+预测，需要保留一下
            self.meta_df = self.data.copy()

        # 2) 遍历其他策略(含NLP也算预测模型, 视需求而定)
        for strat_name, strat_obj in self.strategies.items():
            # 如果是 MetaModel，本示例就不再单独回测资金曲线
            if isinstance(strat_obj, MetaModelStrategy):
                logging.info(f"跳过单独回测 MetaModel，已在上面生成预测值。")
                continue

            logging.info(f"回测策略: {strat_name}")
            eq_curve = self._run_single_strategy(strat_name, strat_obj)
            self.equity_curves[strat_name] = eq_curve

        # 3) 画所有资金曲线对比
        self._plot_comparison()

        # 如果生成了 meta_predicted_close，就画预测K线
        if self.meta_model_used and hasattr(self, 'meta_df'):
            logging.info("绘制 MetaModel 的 预测K线 vs 实际K线 ...")
            self._plot_meta_prediction(self.meta_df)

    def _run_single_strategy(self, strat_name, strategy):
        """
        针对单一策略执行回测。此处的策略为"交易策略" (非预测模型)。
        """
        df = self.data.copy()
        # 用真实 close 来计算当天涨跌
        df['return'] = df['close'].pct_change().fillna(0)

        capital = self.initial_capital
        position_list = []
        equity_curve = pd.Series(index=df.index, dtype=float)

        # 1) 获取该交易策略的买卖信号/仓位
        # 关键：让交易策略基于 'meta_predicted_close' 做判断 => 需要策略内使用这个字段
        # => 要保证策略.generate_signals 里写死 df['close'] => 改成 df['meta_predicted_close'] or 由策略自行判断
        strategy_output = strategy.generate_signals(df)

        # 2) 计算资金曲线
        for i in range(len(df)):
            if i == 0:
                position_list.append(0.0)
                equity_curve.iloc[i] = capital
                continue

            signal_last = strategy_output.iloc[i-1] if not pd.isna(strategy_output.iloc[i-1]) else 0.0
            position = signal_last
            position_list.append(position)

            # 当日的资金变动仍基于真实行情涨跌(df['return'])
            daily_return = position_list[-2] * df['return'].iloc[i]
            capital *= (1 + daily_return)
            equity_curve.iloc[i] = capital

        return equity_curve

    def _plot_comparison(self):
        """
        绘制所有策略的资金曲线
        """
        plt.figure(figsize=(10,6))
        for strat_name, eq_curve in self.equity_curves.items():
            plt.plot(eq_curve, label=strat_name)
        plt.title("Strategies Comparison (Using MetaModel Predictions)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

        # 打印最终收益率
        for strat_name, eq_curve in self.equity_curves.items():
            final_equity = eq_curve.iloc[-1]
            total_return = (final_equity / self.initial_capital - 1) * 100
            print(f"{strat_name} 最终收益率: {total_return:.2f}%")

    def _plot_meta_prediction(self, df):
        """
        绘制K线 + meta_predicted_close线
        """
        # 若 date 列存在，就转Datetime并设为索引
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # 重命名
        df_for_plot = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })

        # 周线重采样(可注释)
        df_week = df_for_plot.resample('W').agg({
            'Open':'first',
            'High':'max',
            'Low':'min',
            'Close':'last',
            'Volume':'sum',
            'meta_predicted_close':'last'
        }).dropna(subset=['Open','High','Low','Close'])

        df_week.reset_index(inplace=True)
        df_week['date_num'] = mdates.date2num(df_week['trade_date'])
        ohlc_data = df_week[['date_num','Open','High','Low','Close']].values

        fig, ax = plt.subplots(figsize=(12,6))

        # 画K线
        candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='red', colordown='green')

        # 画预测值曲线
        if 'meta_predicted_close' in df_week.columns:
            valid = ~df_week['meta_predicted_close'].isna()
            x_pred = df_week.loc[valid,'date_num']
            y_pred = df_week.loc[valid,'meta_predicted_close']
            ax.plot(x_pred, y_pred, '--b', label='Meta Predicted')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        ax.set_title("MetaModel: Predicted vs Real K Line (Weekly)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.show()
