"""
train_transformer_ebs.py

使用 PyTorch 的 nn.TransformerEncoder 做一个最简单的回归:
- 输入: [lookback天, feature_dim=5]
- 输出: 下一日收盘价(实数)
并将模型保存到 transformer_model_state_dict.pt
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 用于显示训练进度条

def load_ebs_data(csv_path):
    """
    加载光大证券数据
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

def create_dataset(df, lookback=60):
    """
    构造回归数据集:
    X: (样本数, lookback, feature_dim=5)
    y: (样本数,)  -- 对应“下一日收盘价”
    """
    data = df[['open', 'high', 'low', 'close', 'vol']].values
    closes = df['close'].values

    X_list, y_list = [], []
    for i in range(len(data) - lookback - 1):
        # 输入: 前 lookback 天的 [open,high,low,close,vol]
        X_list.append(data[i:i+lookback])
        # 输出: 第 i+lookback 天的收盘价 (相当于"下一日")
        y_list.append(closes[i + lookback])

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    return X_arr, y_arr

# ------------------------- (可选) 数据标准化示例 ------------------------------------
# from sklearn.preprocessing import MinMaxScaler
#
# # 示例: 针对 X 的5个特征 统一缩放
# scaler_X = MinMaxScaler()
# # 针对单列 y (收盘价) 的缩放
# scaler_y = MinMaxScaler()
#
# # 训练前:
# X_reshaped = X_arr.reshape(-1, 5)  # (总样本数*lookback, 5)
# X_scaled = scaler_X.fit_transform(X_reshaped).reshape(-1, lookback, 5)
# y_scaled = scaler_y.fit_transform(y_arr.reshape(-1,1)).ravel()
#
# # 训练结束后预测:
# y_pred_scaled = model(X_test_t)   # 这是缩放后的预测值
# y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().numpy())
# -----------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """
    简单的 Transformer, 现在做回归输出，去掉 Sigmoid
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_fc = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
        # 注意：不再使用 Sigmoid，因为我们做回归

    def forward(self, x):
        """
        x: shape (batch_size, lookback, 5)
        """
        x = self.input_fc(x)               # -> (batch_size, lookback, d_model)
        x = self.transformer_encoder(x)    # -> (batch_size, lookback, d_model)
        x_last = x[:, -1, :]               # 取最后时刻的输出
        out = self.output_fc(x_last)       # -> (batch_size, 1)
        return out                         # 实数回归

def train_model(model, X_train, y_train, criterion, optimizer, epochs=10):
    """
    训练回归模型
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)       # shape: (batch_size,1)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # 计算 MAE 作为参考
        with torch.no_grad():
            mae = (pred - y_train).abs().mean().item()
        print(f"Epoch {epoch+1}/{epochs}, Loss(MSE)={loss.item():.4f}, MAE={mae:.4f}")

def evaluate_model(model, X_test, y_test, criterion):
    """
    测试回归模型
    """
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = criterion(pred, y_test).item()
        mae = (pred - y_test).abs().mean().item()
    print(f"Test MSE={mse:.4f}, MAE={mae:.4f}")
    return mse, mae, pred

if __name__ == "__main__":
    # 1) 读取数据
    csv_path = "C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\data\\601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    # 2) 创建回归数据集
    lookback = 20
    X, y = create_dataset(df_ebs, lookback=lookback)

    # 3) 划分训练/测试集 (80%:20%)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4) 转换为张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # shape: (batch, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # (可选) 对X,y做归一化，这里省略，见上方示例注释

    # 5) 初始化模型 (回归)
    model = SimpleTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6) 训练模型
    print("Start training (Regression for next-day close)...")
    train_model(model, X_train_t, y_train_t, criterion, optimizer, epochs=10)

    # 7) 测试模型
    print("Evaluating model on test set...")
    mse, mae, pred_test = evaluate_model(model, X_test_t, y_test_t, criterion)

    # 如果你对y做了scaler_y的inverse_transform，这里需要先把pred_test还原
    #   y_pred = scaler_y.inverse_transform(pred_test.numpy())
    #   y_real = scaler_y.inverse_transform(y_test_t.numpy())

    # 8) 保存模型
    model_dir = os.path.join("C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\utils\\models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "transformer_model_state_dict.pt"))
    print(f"Regression Transformer saved to {model_dir}/transformer_model_state_dict.pt")
