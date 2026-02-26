#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 LSTM 预测单支股票“下一日收盘价”
—————————————————————————————————————
• 读取 CSV（示例：601788.SH）
• 构造 look-back=120 天样本 (X, y)
• 训练两层 LSTM → Dense(1) 模型
• 评估、保存模型与 scaler
• 可视化：Loss 曲线 / 真实 vs 预测 / 残差分布
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# 数据加载
def load_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[:, ['open', 'high', 'low', 'close', 'vol']].dropna()
    df = df.sort_index()
    return df

# 数据集构造
def make_dataset(df: pd.DataFrame, lookback: int = 60):
    feature_cols = ['open', 'high', 'low', 'close', 'vol']
    feat_values = df[feature_cols].values.astype('float32')

    scaler_x = MinMaxScaler((0, 1))
    feat_scaled = scaler_x.fit_transform(feat_values)

    X, y = [], []
    for i in range(len(df) - lookback - 5 + 1):
        X.append(feat_scaled[i: i + lookback])
        y.append(np.mean(feat_scaled[i + lookback: i + lookback + 5, 3]))  # 第3列的均值（close）

    return np.array(X), np.array(y), scaler_x

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)  # 线性输出
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# 反归一化工具
def inverse_close(scaled_close_vec, scaler_x):
    dummy = np.zeros((len(scaled_close_vec), 5))
    dummy[:, 3] = scaled_close_vec  # 仅填充close，其余列占位
    return scaler_x.inverse_transform(dummy)[:, 3]

# 主流程
if __name__ == "__main__":
    # 配置路径
    csv_path = "data/601788_SH.csv"  # 修改为你自己的路径
    model_dir = "utils/models"
    os.makedirs(model_dir, exist_ok=True)

    # 加载数据
    df = load_price_data(csv_path)
    lookback = 60
    X, y, scaler_x = make_dataset(df, lookback)

    # 时间顺序划分80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 构建并训练模型
    model = build_lstm_model(input_shape=(lookback, X.shape[-1]))
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        verbose=2
    )

    # 评估
    test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MSE = {test_mse:.4f}")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_true = inverse_close(y_test, scaler_x)
    y_pred = inverse_close(y_pred_scaled, scaler_x)
    print(f"Test MAE = {mean_absolute_error(y_true, y_pred):.4f}")

    # 保存
    model_path = os.path.join(model_dir, "lstm_model.h5")
    scaler_path = os.path.join(model_dir, "lstm_close_scaler.pkl")
    save_model(model, model_path)
    joblib.dump(scaler_x, scaler_path)
    print(f"\n模型已保存至: {model_path}")
    print(f"Scaler 已保存至: {scaler_path}")

    # 可视化
    # 5-1 训练 & 验证损失
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train MSE")
    plt.plot(history.history["val_loss"], label="Val MSE")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5-2 真实价 vs 预测价（测试集）
    test_dates = df.index[-len(y_true):]
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_true, label="Actual Close")
    plt.plot(test_dates, y_pred, label="Predicted Close")
    plt.title("Actual vs Predicted Close (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5-3 残差分布
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.title("Prediction Residuals")
    plt.xlabel("Actual − Predicted (Price)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # （可选）方向准确率
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    acc = (direction_true == direction_pred).mean()
    print(f"Direction Accuracy = {acc:.2%}")