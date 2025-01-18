import time
import pandas as pd
from config.config import BROKER_CONFIG

class LiveTrader:
    def __init__(self, strategy, broker_config=BROKER_CONFIG):
        self.strategy = strategy
        self.broker_config = broker_config
        self.current_position = 0.0

        print(f"[LiveTrader] 已连接到券商: {broker_config['broker_name']} (模拟环境)")

    def get_realtime_data(self):
        simulated_data = {
            "open": 10.0,
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "vol": 100000
        }
        return simulated_data

    def execute_trade(self, target_position):
        diff = target_position - self.current_position
        if abs(diff) < 1e-3:
            print("仓位变化极小，无需操作")
        elif diff > 0:
            print(f"【买入】增加仓位: {diff:.2f}")
            self.current_position += diff
        else:
            print(f"【卖出】减少仓位: {-diff:.2f}")
            self.current_position += diff

    def start_trading(self):
        print("开始模拟实盘交易... (Ctrl+C 终止)")
        try:
            while True:
                realtime_data = self.get_realtime_data()
                df = pd.DataFrame([realtime_data])

                signal_series = self.strategy.generate_signals(df)
                target_position = signal_series.iloc[-1] if not signal_series.empty else 0.0

                # 限制在0~1之间
                target_position = max(0.0, min(target_position, 1.0))

                self.execute_trade(target_position)
                print(f"当前持仓比例: {self.current_position:.2f}")

                time.sleep(5)
        except KeyboardInterrupt:
            print("停止实盘(模拟)交易。")
