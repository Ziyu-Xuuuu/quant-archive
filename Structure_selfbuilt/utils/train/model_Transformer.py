#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_transformer_ebs.py

使用 PyTorch 的 nn.TransformerEncoder 做一个最简单的回归:
- 输入: [lookback天, feature_dim=5]
- 输出: 下一日收盘价(实数)
训练结束后可视化:
1. 训练 Loss 曲线
2. 测试集 实际价 vs 预测价
3. 残差直方图

模型权重保存在 transformer_model_state_dict.pt
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm               # 进度条
import matplotlib.pyplot as plt     # <<< NEW
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# ------------------------- 数据加载 ------------------------- #
def load_ebs_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

# ------------------------- 数据集构造 ------------------------- #
def create_dataset(df, lookback=60):
    data_raw = df[['open', 'high', 'low', 'close', 'vol']].values
    closes = df['close'].values
    scaler_x = MinMaxScaler((0, 1))
    feat_scaled = scaler_x.fit_transform(data_raw)

    X_list, y_list = [], []
    for i in range(len(feat_scaled) - lookback - 5):
        X_list.append(feat_scaled[i:i+lookback])
        y_list.append(np.mean(feat_scaled[i + lookback: i + lookback + 5, 3]))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ------------------------- 模型 ------------------------- #
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_fc = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)            # (B, L, d_model)
        x = self.transformer_encoder(x) # (B, L, d_model)
        x_last = x[:, -1, :]            # 取最后一个时间步
        return self.output_fc(x_last)   # (B, 1)

# ------------------------- 训练 / 评估 ------------------------- #
loss_history = []     # <<< NEW：用来记录每 epoch 的 loss

def train_model(model, X_train, y_train, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # 记录 & 打印
        loss_history.append(loss.item())            # <<< NEW
        mae = (pred - y_train).abs().mean().item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}, MAE={mae:.4f}")

def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse  = criterion(pred, y_test).item()
        mae  = (pred - y_test).abs().mean().item()
    print(f"Test MSE={mse:.4f}, MAE={mae:.4f}")
    return mse, mae, pred

# ------------------------- 主流程 ------------------------- #
if __name__ == "__main__":
    # 1) 读取数据
    csv_path = r"C:\Users\user\Documents\GitHub\trader\Stock_Trade\data\601788_SH.csv"
    df_ebs   = load_ebs_data(csv_path)

    # 2) 创建数据集
    lookback = 20
    X, y = create_dataset(df_ebs, lookback)

    # 3) 划分训练 / 测试 (时间顺序 80:20)
    split   = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4) 转为张量
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)
    X_test_t  = torch.tensor(X_test)
    y_test_t  = torch.tensor(y_test).view(-1, 1)

    # 5) 初始化模型
    model     = SimpleTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6) 训练
    print("Start training...")
    train_model(model, X_train_t, y_train_t, criterion, optimizer, epochs=10)

    # 7) 测试
    print("\nEvaluating...")
    mse, mae, pred_test = evaluate_model(model, X_test_t, y_test_t, criterion)

    # 8) 保存模型
    model_dir = r"C:\Users\user\Documents\GitHub\trader\Stock_Trade\utils\models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "transformer_model_state_dict.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # =========================================================
    # ====================  可 视 化  ==========================
    # =========================================================
    # 1) Loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title("Training Loss (MSE) per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.show()

    # 2) 实际价 vs 预测价（测试集）
    y_true = y_test_t.numpy().flatten()
    y_pred = pred_test.numpy().flatten()

    test_dates = df_ebs.index[-len(y_true):]   # 与测试集长度对应的日期
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_true, label="Actual Close")
    plt.plot(test_dates, y_pred, label="Predicted Close")
    plt.title("Actual vs Predicted Close Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) 残差直方图
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.title("Prediction Residuals (Actual - Predicted)")
    plt.xlabel("Price Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
