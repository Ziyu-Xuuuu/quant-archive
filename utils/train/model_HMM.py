#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hmm_ebs.py  (带归一化 + 未来 5 日均值可视化)

• 将 5 维特征统一 MinMaxScaler 到 (0,1)
• 计算 future5_mean_close (同样缩放后) 作参考曲线
• 训练 2-state 高斯 HMM
• 三幅可视化:
  1. EM 迭代 log-likelihood
  2. 收盘价 + 隐藏状态着色 + 5 日均值
  3. 隐藏状态计数
• 保存模型和 scaler
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM
import joblib

# ------------------------------------------------------------
# 1. 数据加载 + 归一化 + future-5-mean
# ------------------------------------------------------------
def make_dataset(df: pd.DataFrame, lookback: int = 60):
    """
    返回:
        feat_scaled : (N, 5)  — 归一化后的 open/high/low/close/vol
        future5_mean_close_scaled : (N,) — 归一化后 close 的未来 5 日均值
        scaler_x :  fitted MinMaxScaler
    说明:
        第 i 行的 future5_mean_close = mean(close[i : i+5])
        末尾不足 5 行的地方将被截去, 确保对齐
    """
    feature_cols = ['open', 'high', 'low', 'close', 'vol']
    feat_values  = df[feature_cols].values.astype('float32')

    scaler_x = MinMaxScaler((0, 1))
    feat_scaled = scaler_x.fit_transform(feat_values)

    closes_scaled = feat_scaled[:, 3]             # close 列 (缩放后)
    means = []
    for i in range(len(closes_scaled) - 5 + 1):
        means.append(closes_scaled[i : i + 5].mean())
    means = np.array(means, dtype=np.float32)

    # 需把 feature 矩阵截成与 means 一致长度
    feat_scaled = feat_scaled[: len(means)]

    return feat_scaled, means, scaler_x

# ------------------------------------------------------------
# 2. 可视化工具
# ------------------------------------------------------------
def plot_hidden_states(dates, close_prices, hidden_states,
                       future5_mean, n_states):
    # 2-1 收盘价 + 状态着色
    plt.figure(figsize=(12, 5))
    for s in range(n_states):
        mask = hidden_states == s
        plt.plot(dates[mask], close_prices[mask],
                 linestyle="None", marker="o", markersize=3,
                 label=f"State {s}")
    plt.plot(dates, future5_mean, color="black", linewidth=1,
             label="Future-5-Day Mean (scaled)")
    plt.title("Close Price • Hidden State Coloring • 5-Day Mean")
    plt.xlabel("Date"); plt.ylabel("Scaled Close")
    plt.legend(); plt.tight_layout(); plt.show()

    # 2-2 状态计数
    plt.figure(figsize=(6, 4))
    plt.hist(hidden_states, bins=np.arange(-0.5, n_states+0.5, 1), rwidth=0.8)
    plt.xticks(range(n_states))
    plt.title("Hidden-State Counts"); plt.xlabel("State"); plt.ylabel("Freq")
    plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    # —— 路径
    csv_path = r"C:\Users\user\Documents\GitHub\trader\Stock_Trade\data\601788_SH.csv"
    model_dir = r"..\models"
    os.makedirs(model_dir, exist_ok=True)

    # —— 加载原始 DF
    raw_df = (
        pd.read_csv(csv_path, index_col=0, parse_dates=True)
          .loc[:, ['open', 'high', 'low', 'close', 'vol']]
          .dropna()
          .sort_index()
    )

    # —— 构造数据集 (归一化 + future5 均值)
    feat_scaled, future5_mean_scaled, scaler_x = make_dataset(raw_df)

    # —— 训练 HMM
    hmm_model = GaussianHMM(n_components=2, covariance_type='full',
                            n_iter=100, random_state=42, verbose=False)
    hmm_model.fit(feat_scaled)
    print("HMM trained on scaled data, shape =", feat_scaled.shape)
    print(f"Log-likelihood: {hmm_model.score(feat_scaled):.2f}")

    # —— 保存模型 & scaler
    joblib.dump(hmm_model, os.path.join(model_dir, "hmm_model.pkl"))
    joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.pkl"))
    print("Model  &  scaler saved to", model_dir)

    # =========================================================
    # ====================   可 视 化   ========================
    # =========================================================
    # 1) EM log-likelihood 曲线
    if hasattr(hmm_model.monitor_, "history") and hmm_model.monitor_.history:
        plt.figure(figsize=(8, 4))
        plt.plot(hmm_model.monitor_.history, marker='o')
        plt.title("EM Log-Likelihood per Iteration")
        plt.xlabel("Iteration"); plt.ylabel("Log-Likelihood")
        plt.tight_layout(); plt.show()
    else:
        print("monitor_.history 不可用，跳过 EM 曲线。")

    # 2) 收盘价 & 隐藏状态着色 & 未来 5 日均值
    hidden_states = hmm_model.predict(feat_scaled)
    dates_trim    = raw_df.index[: len(future5_mean_scaled)]
    close_scaled  = feat_scaled[:, 3]   # 已缩放
    plot_hidden_states(dates_trim, close_scaled,
                       hidden_states, future5_mean_scaled,
                       hmm_model.n_components)
