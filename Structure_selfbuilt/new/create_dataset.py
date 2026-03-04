import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from new.regime_hmm import compute_states_from_df

# 输入和输出文件路径
INPUT_CSV = r"data\601788_SH.csv"
OUTPUT_CSV = r"data\market_states_601788_SH.csv"

def create_sliding_window_dataset(df, N, H):
    """
    构建滑动窗口数据集
    :param df: 数据框
    :param N: 输入特征的窗口大小（过去 N 天）
    :param H: 预测未来 H 天收益
    :return: 输入 X 和标签 y
    """
    X = []
    y = []
    
    # 生成滑动窗口数据
    for i in range(N, len(df) - H + 1):  # 避免越界
        x_t = df.iloc[i - N:i][['hmm_p0', 'hmm_p1', 'hmm_p2', 'hmm_p3', 'hmm_p4']].values  # 输入数据
        X.append(x_t)
        
        # 计算未来 H 天的收益
        fwd_ret = np.log(df['close'].iloc[i + H - 1]) - np.log(df['close'].iloc[i - 1])  # H 天的收益
        y.append(1 if fwd_ret > 0 else 0)  # 如果收益大于 0，则为 1，否则为 0

    # 打印 X 和 y 的样本检查
    print(f"Generated {len(X)} samples")
    print(f"Sample of X values (first sample): {X[0] if len(X) > 0 else 'Empty'}")
    print(f"Sample of y values (first 30): {y[:30]}")

    return np.array(X), np.array(y)


def main():
    # 1. 读已有的 CSV
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)

    # 2. 把索引变成一列，并改成 compute_states_from_df 需要的 'date'
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})  # trade_date -> date

    # 3. 调用状态机
    df_states = compute_states_from_df(df_reset)

    # 4. 创建滑动窗口数据集
    N = 30  # 过去30天作为输入
    H = 5   # 预测未来5天的收益
    X, y = create_sliding_window_dataset(df_states, N, H)

    # 5. 检查 y 和 df_states 长度是否匹配
    print(f"Length of df_states: {len(df_states)}")
    print(f"Length of y: {len(y)}")

    # 调整 y 的长度，确保它与 df_states 匹配
    df_states = df_states.iloc[N:]  # 只保留从第 N 行开始的数据
    print(f"Length of df_states after slicing: {len(df_states)}")

    # 如果 y 的长度不匹配 df_states，截断 y
    if len(df_states) != len(y):
        print(f"Warning: Length of df_states ({len(df_states)}) and y ({len(y)}) do not match.")
        # 截断 y 使其与 df_states 匹配
        y = y[:len(df_states)]

    # 6. 打印 y 内容，确认其正确性
    print(f"Sample of y values (first 30): {y[:30]}")  # 打印 y 的前 30 个值进行检查

    # 7. 将 y 作为标签列添加到 df_states
    df_states["y"] = y  # 将 y 作为标签列添加到 df_states
    print(f"Columns in df_states: {df_states.columns.tolist()}")  # 打印 df_states 的列，确认 y 列是否已添加

    # 8. 保存到 CSV
    cols_out = [
        "ts_code", "open", "high", "low", "close", "vol",
        "trend_regime", "vol_regime", "state", "y"  # 添加 'y' 列
    ]
    if "hmm_state" in df_states.columns:
        cols_out += ["hmm_state", "hmm_state_label"]

    df_states[cols_out].to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"Saved to: {OUTPUT_CSV}")
    print(df_states[cols_out].tail())




