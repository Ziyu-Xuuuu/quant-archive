import pandas as pd
from regime_hmm import compute_states_from_df

# 先用 data 里的 601788 做测试
INPUT_CSV = r"data\601788_SH.csv"
OUTPUT_CSV = r"data\market_states_601788_SH.csv"

def main():
    # 1. 读已有的 CSV（你之前 save_data_to_csv 存出来的）
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)

    # 此时索引一般叫 trade_date，列是：ts_code, open, high, low, close, vol
    # 2. 把索引变成一列，并改成 compute_states_from_df 需要的 'date'
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})  # trade_date -> date

    # 3. 调用状态机
    df_states = compute_states_from_df(df_reset)

    # 4. 把 date 设回索引，索引名仍然叫 trade_date，兼容你原来的习惯
    df_states = df_states.set_index("date")
    df_states.index.name = "trade_date"

    # 5. 保存一个新的 CSV，看看结果
    cols_out = [
        "ts_code", "open", "high", "low", "close", "vol",
        "trend_regime", "vol_regime", "state"
    ]
    if "hmm_state" in df_states.columns:
        cols_out += ["hmm_state", "hmm_state_label"]

    df_states[cols_out].to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print("saved to:", OUTPUT_CSV)
    print(df_states[cols_out].tail())


if __name__ == "__main__":
    main()