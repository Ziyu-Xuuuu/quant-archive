from utils.data_fetcher import DataFetcher

fetcher = DataFetcher()
df_ebs = fetcher.get_historical_data(ts_code="601788.SH", start_date="20190101", end_date="20230101")

# 把数据保存到 CSV, 例如 "601788_SH.csv"
fetcher.save_data_to_csv(df_ebs, "601788_SH.csv")
