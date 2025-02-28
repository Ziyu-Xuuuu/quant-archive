"""
train_hmm_ebs.py

使用 hmmlearn 训练一个高斯HMM, 并明确使用5维特征: [open, high, low, close, vol].
最终保存到 models/hmm_model.pkl
"""
import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib

def load_ebs_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # 明确使用5个特征: open, high, low, close, vol
    df = df[['open','high','low','close','vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    # 1) 读取数据
    csv_path = "C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\data\\601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    # 2) 构造训练数据: 5 维特征
    data = df_ebs[['open','high','low','close','vol']].values  # shape = (N, 5)

    # 3) 训练一个2状态的HMM (你可调n_components=3,4,...)
    hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)

    hmm_model.fit(data)
    print("HMM has been trained with shape =", data.shape)

    # 4) 评估(无监督: 只能看log likelihood)
    log_likelihood = hmm_model.score(data)
    print(f"HMM log likelihood: {log_likelihood:.2f}")

    # 5) 保存到 pkl
    os.makedirs("../models", exist_ok=True)
    joblib.dump(hmm_model, "../models/hmm_model.pkl")
    print("HMM model has been saved to ../models/hmm_model.pkl")
