import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# 预测模型
from strategies.meta_model_strategy import MetaModelStrategy


class Backtester:
    def __init__(self, strategies: dict, data: pd.DataFrame, initial_capital=100000.0):
        """
        strategies: {策略名: 策略对象}
        data: 原始行情 DataFrame，含 [open, high, low, close, vol] 等
        initial_capital: 初始资金
        """
        self.strategies = strategies
        self.data = data.copy()
        self.initial_capital = float(initial_capital)
        self.equity_curves = {}
        self.meta_model_used = False
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def run_backtest(self):
        """
        流程:
        1) 如果有 MetaModel 策略 => 先用它对 self.data 生成 'meta_predicted_close'
        2) 其他策略基于 self.data 生成信号 => 计算资金曲线
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
            self.data["meta_predicted_close"] = meta_output
            self.meta_model_used = True
            self.meta_df = self.data.copy()

        # 2) 遍历其他策略
        for strat_name, strat_obj in self.strategies.items():
            if isinstance(strat_obj, MetaModelStrategy):
                logging.info("跳过单独回测 MetaModel，已在上面生成预测值。")
                continue

            logging.info(f"回测策略: {strat_name}")
            eq_curve = self._run_single_strategy(strat_name, strat_obj)
            self.equity_curves[strat_name] = eq_curve

        # 3) 画资金曲线对比
        self._plot_comparison()

        # 4) 画Meta预测K线（如果有）
        if self.meta_model_used and hasattr(self, "meta_df"):
            logging.info("绘制 MetaModel 的 预测K线 vs 实际K线 ...")
            self._plot_meta_prediction(self.meta_df)

    def _run_single_strategy(self, strat_name, strategy):
        """
        针对单一策略执行回测。
        """
        df = self.data.copy()

        # 用真实 close 计算日收益
        if "close" not in df.columns:
            raise KeyError("Backtester: data 缺少 close 列")
        df["return"] = df["close"].pct_change().fillna(0)

        capital = self.initial_capital
        position_list = []
        equity_curve = pd.Series(index=df.index, dtype=float)

        # 1) 策略输出仓位/信号序列（长度=len(df)）
        strategy_output = strategy.generate_signals(df)

        # 2) 计算资金曲线
        for i in range(len(df)):
            if i == 0:
                position_list.append(0.0)
                equity_curve.iloc[i] = capital
                continue

            signal_last = strategy_output.iloc[i - 1] if not pd.isna(strategy_output.iloc[i - 1]) else 0.0
            position = float(signal_last)
            position_list.append(position)

            daily_return = position_list[-2] * float(df["return"].iloc[i])
            capital *= (1 + daily_return)
            equity_curve.iloc[i] = capital

        return equity_curve

    def _plot_comparison(self):
        """
        绘制所有策略的资金曲线
        """
        plt.figure(figsize=(10, 6))
        for strat_name, eq_curve in self.equity_curves.items():
            plt.plot(eq_curve, label=strat_name)
        plt.title("Strategies Comparison (Using MetaModel Predictions)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

        # 打印最终收益率
        for strat_name, eq_curve in self.equity_curves.items():
            final_equity = float(eq_curve.iloc[-1])
            total_return = (final_equity / self.initial_capital - 1) * 100
            print(f"{strat_name} 最终收益率: {total_return:.2f}%")

    def _plot_meta_prediction(self, df: pd.DataFrame):
        """
        绘制K线 + meta_predicted_close线
        兼容索引为 trade_date/date，或存在 trade_date/date 列的情况
        """
        df = df.copy()

        # 1) 统一 datetime 索引
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.set_index("trade_date")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df.sort_index()
        df = df.dropna(subset=["open", "high", "low", "close"], how="any")

        # 2) 重命名绘图字段
        df_for_plot = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        })

        # 3) 周线重采样
        agg_map = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Volume" in df_for_plot.columns:
            agg_map["Volume"] = "sum"
        if "meta_predicted_close" in df_for_plot.columns:
            agg_map["meta_predicted_close"] = "last"

        df_week = df_for_plot.resample("W").agg(agg_map)
        df_week = df_week.dropna(subset=["Open", "High", "Low", "Close"])

        df_week = df_week.reset_index()
        dt_col = df_week.columns[0]  # reset_index 后第一列就是日期列
        df_week["date_num"] = mdates.date2num(df_week[dt_col])

        ohlc_data = df_week[["date_num", "Open", "High", "Low", "Close"]].values

        fig, ax = plt.subplots(figsize=(12, 6))

        # 画K线
        candlestick_ohlc(ax, ohlc_data, width=0.6, colorup="red", colordown="green")

        # 画预测值曲线
        if "meta_predicted_close" in df_week.columns:
            valid = ~df_week["meta_predicted_close"].isna()
            x_pred = df_week.loc[valid, "date_num"]
            y_pred = df_week.loc[valid, "meta_predicted_close"]
            ax.plot(x_pred, y_pred, "--b", label="Meta Predicted")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        ax.set_title("MetaModel: Predicted vs Real K Line (Weekly)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.show()

