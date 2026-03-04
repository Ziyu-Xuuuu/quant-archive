import pandas as pd
import numpy as np

# === 1. 路径 ===
INPUT_CSV = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\market_states_601788_SH.csv"
OUTPUT_CSV = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\market_states_601788_SH_with_labels.csv"

# === 2. 读取数据 ===
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values("trade_date").reset_index(drop=True)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# === 3. 生成未来收益标签 ===
H = 5  # 预测未来5天

df["fwd_ret"] = np.log(df["close"]).shift(-H) - np.log(df["close"])
df["y"] = (df["fwd_ret"] > 0).astype(int)

print("\nLabel distribution:")
print(df["y"].value_counts(dropna=False))

# === 4. 删除未来无标签部分 ===
df = df.dropna(subset=["fwd_ret"]).reset_index(drop=True)

print("\nAfter dropping NaN labels:", df.shape)

# === 5. 确保 y 列在列名中 ===
print("Columns before saving:", df.columns.tolist())  # 打印列名确认是否包含 y 列

# === 6. 选择要保存的列 ===
cols_out = [
    "ts_code", "open", "high", "low", "close", "vol",
    "trend_regime", "vol_regime", "state", "y",  # 包括 y 列
    "hmm_state", "hmm_state_label",  # 包含 HMM 状态列
    "hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"  # 包含 HMM 概率列
]

# === 7. 保存数据 ===
df[cols_out].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved to: {OUTPUT_CSV}")
print(df[cols_out].tail())  # 打印最后几行数据确认保存


