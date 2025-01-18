"""
train_hmm_ebs.py

使用 hmmlearn 训练一个高斯HMM, 模拟将光大证券(601788.SH)行情分为2种隐藏状态.
最终保存到 models/hmm_model.pkl
"""
import os
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
import joblib

def load_ebs_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[['open','high','low','close','vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    csv_path = "/Stock_Trade/data/601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    # 构造训练数据 (无监督, 只需要 X)
    # 例如: X = [ [open, high, low, close, vol], ...]
    data = df_ebs[['open','high','low','close','vol']].values

    # 训练一个2状态的HMM (你可调 n_components=3,4,...)
    hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=100)

    # fit
    hmm_model.fit(data)

    # 评估(无监督没有固定的label, 可以看score)
    log_likelihood = hmm_model.score(data)
    print(f"HMM log likelihood: {log_likelihood:.2f}")

    # 保存到 pkl
    os.makedirs("../models", exist_ok=True)
    joblib.dump(hmm_model, "../models/hmm_model.pkl")
    print("HMM model has been saved to models/hmm_model.pkl")
