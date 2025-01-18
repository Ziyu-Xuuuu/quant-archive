# data_fetcher.py
import tushare as ts
import pandas as pd
import os
from config.config import TUSHARE_TOKEN


class DataFetcher:
    def __init__(self):
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()

    def get_historical_data(self, ts_code, start_date, end_date, adj='qfq'):
        """
        从 Tushare 获取 A 股历史数据
        """
        df_daily = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df_adj = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)

        # 合并复权因子
        df = pd.merge(df_daily, df_adj, on=["ts_code", "trade_date"], how="left")
        df["adj_factor"] = df["adj_factor"].fillna(method="ffill")

        if adj == 'qfq':
            # 前复权
            last_factor = df["adj_factor"].iloc[-1]
            df["open"] = df["open"] * df["adj_factor"] / last_factor
            df["high"] = df["high"] * df["adj_factor"] / last_factor
            df["low"] = df["low"] * df["adj_factor"] / last_factor
            df["close"] = df["close"] * df["adj_factor"] / last_factor

        # 处理时间索引
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.sort_values('trade_date', inplace=True)
        df.set_index('trade_date', inplace=True)

        # 只保留需要的列
        df = df[['ts_code', 'open', 'high', 'low', 'close', 'vol']]
        return df

    def save_data_to_csv(self, df, filename):
        if not os.path.exists('data'):
            os.makedirs('data')
        df.to_csv(os.path.join('data', filename))
        print(f"数据已保存至 data/{filename}")

    def read_data_from_csv(self, filename):
        path = os.path.join('data', filename)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
