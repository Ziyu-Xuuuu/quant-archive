"""
train_transformer_ebs.py

使用 PyTorch 自带的 nn.TransformerEncoder 做一个简单的时间序列预测:
- 输入: [lookback天, feature_dim=5]
- 输出: 1个二分类概率(是否上涨)
并将模型保存为 models/transformer_model_state_dict.pt
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

def create_dataset(df, lookback=20):
    """
    构造特征数据集
    """
    data = df[['open', 'high', 'low', 'close', 'vol']].values
    closes = df['close'].values

    X_list, y_list = [], []
    for i in range(len(data) - lookback - 1):
        X_list.append(data[i:i+lookback])
        # 下一日是否上涨
        label = 1.0 if closes[i+lookback] > closes[i+lookback-1] else 0.0
        y_list.append(label)
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr

class SimpleTransformer(nn.Module):
    """
    简化的 Transformer
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_fc = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x_last = x[:, -1, :]  # 取最后时间步
        out = self.output_fc(x_last)
        return self.sigmoid(out)

def train_model(model, X_train, y_train, criterion, optimizer, epochs=10):
    """
    模型训练
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)  # [batch, 1]
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # 训练精度
        with torch.no_grad():
            preds_label = (pred > 0.5).float()
            acc = (preds_label == y_train).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}, Train Accuracy={acc:.4f}")

def evaluate_model(model, X_test, y_test, criterion):
    """
    模型测试
    """
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        loss = criterion(pred, y_test).item()
        pred_label = (pred > 0.5).float()
        acc = (pred_label == y_test).float().mean().item()
    print(f"Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")
    return loss, acc

if __name__ == "__main__":
    # 数据路径
    csv_path = "C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\data\\601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    # 创建数据集
    lookback = 20
    X, y = create_dataset(df_ebs, lookback=lookback)

    # 数据划分
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 初始化模型
    model = SimpleTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Start training...")
    train_model(model, X_train_t, y_train_t, criterion, optimizer, epochs=10)

    # 测试模型
    print("Evaluating model...")
    loss, acc = evaluate_model(model, X_test_t, y_test_t, criterion)

    # 保存模型
    model_dir = os.path.join("C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\utils\\models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "transformer_model_state_dict.pt"))
    print(f"Model saved to {model_dir}/transformer_model_state_dict.pt")
