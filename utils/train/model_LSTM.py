"""
train_lstm_ebs.py

演示使用 Keras (TensorFlow) 训练一个最简单的 LSTM 模型，预测光大证券(601788.SH) 下一日涨跌。
并将训练好的模型保存到 models/lstm_model.h5。
"""
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_ebs_data(csv_path):
    """
    加载光大证券历史数据, 返回 DataFrame (含 open, high, low, close, vol).
    这里假设 csv_path = 'data/601788_SH.csv'
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # 确保包含列: open, high, low, close, vol
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

def build_lstm_model(input_shape):
    """
    构建一个简单的 LSTM 模型:
    input_shape = (lookback, feature_dim)
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))  # 输出(0~1), 表示上涨概率
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

def create_dataset(df, lookback=20):
    """
    构造训练数据:
    X: (样本数, lookback, feature_dim=5)
    y: (样本数,)  -- 0或1表示下一日是否上涨
    这里仅做示例：若 df[i+1].close > df[i].close -> 1，否则 0
    """
    data = df[['open','high','low','close','vol']].values
    closes = df['close'].values
    X_list, y_list = [], []
    for i in range(len(data) - lookback - 1):
        X_list.append(data[i : i+lookback])
        # 判断下一日 (i+lookback) 是否上涨
        if closes[i+lookback] > closes[i+lookback-1]:
            label = 1.0
        else:
            label = 0.0
        y_list.append(label)
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr

if __name__ == "__main__":
    csv_path = "/Stock_Trade/data/601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    lookback = 20
    X, y = create_dataset(df_ebs, lookback=lookback)

    # 划分训练/测试 (80% 训练, 20% 测试)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建模型
    model = build_lstm_model(input_shape=(lookback, 5))

    # 训练
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # 评估
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

    # 保存模型
    os.makedirs("../models", exist_ok=True)
    save_model(model, "../models/lstm_model.h5")
    print("LSTM model has been saved to models/lstm_model.h5")
