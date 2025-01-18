import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, strategies: dict, data: pd.DataFrame, initial_capital=100000.0):
        self.strategies = strategies
        self.data = data
        self.initial_capital = initial_capital
        self.equity_curves = {}

    def run_backtest(self):
        for strat_name, strat_obj in self.strategies.items():
            print(f"回测策略: {strat_name}")
            equity_curve = self._run_single_strategy(strat_obj)
            self.equity_curves[strat_name] = equity_curve

        self._plot_comparison()

    def _run_single_strategy(self, strategy):
        df = self.data.copy()
        df['return'] = df['close'].pct_change().fillna(0)

        capital = self.initial_capital
        position_list = []
        equity_curve = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            current_data = df.iloc[:i]
            if len(current_data) < 2:
                # 初期数据不足，默认空仓
                position = 0.0
                equity_curve.iloc[i] = capital
                position_list.append(position)
                continue

            # 基于当前历史数据计算策略信号(不含第i行)
            signal_series = strategy.generate_signals(current_data)
            # 最新信号 (上一根bar)
            latest_signal = signal_series.iloc[-1] if not signal_series.empty else 0.0

            # 限制在 0~1
            latest_signal = max(0.0, min(latest_signal, 1.0))

            # 今天(第i行)的收益 = 昨日持仓 * today_return
            daily_return = position_list[-1] * df['return'].iloc[i] if i>0 else 0.0
            capital = capital * (1 + daily_return)

            position_list.append(latest_signal)
            equity_curve.iloc[i] = capital

        return equity_curve

    def _plot_comparison(self):
        plt.figure(figsize=(10, 6))
        for strat_name, eq_curve in self.equity_curves.items():
            plt.plot(eq_curve, label=strat_name)
        plt.title("Strategies Comparison")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.show()

        # 打印最终收益
        for strat_name, eq_curve in self.equity_curves.items():
            final_equity = eq_curve.iloc[-1]
            total_return = (final_equity / self.initial_capital - 1) * 100
            print(f"{strat_name} 最终收益率: {total_return:.2f}%")
