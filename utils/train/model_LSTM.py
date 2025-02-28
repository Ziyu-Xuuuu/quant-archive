import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_ebs_data(csv_path):
    """
    加载光大证券历史数据, 返回 DataFrame (含 open, high, low, close, vol).
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[['open', 'high', 'low', 'close', 'vol']].dropna()
    df.sort_index(inplace=True)
    return df

def build_lstm_model(input_shape):
    """
    构建一个用于回归预测下一日收盘价的 LSTM 模型:
    input_shape = (lookback, feature_dim)
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape))
    # 回归输出层，不要使用 sigmoid/softmax
    model.add(Dense(1))
    # 回归问题常用 'mse' 或 'mae'
    model.compile(loss='mse', optimizer=Adam(0.001))
    return model

def create_dataset(df, lookback=60):
    """
    构造回归训练数据:
    X: (样本数, lookback, feature_dim=5)
    y: (样本数,)  -- 对应“下一日收盘价”

    示例：y[i] = df['close'][i+lookback]，表示第 i+lookback 这个位置的收盘价
    """
    data = df[['open','high','low','close','vol']].values
    closes = df['close'].values

    X_list, y_list = [], []
    # 这里 -1 是为了能拿到 "下一日" 的价格
    for i in range(len(data) - lookback - 1):
        # 取最近 60 天作为一个样本
        X_list.append(data[i : i + lookback])
        # 取下一日的收盘价作为标签
        y_list.append(closes[i + lookback])

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr

if __name__ == "__main__":
    # 1) 加载数据
    csv_path = "C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\data\\601788_SH.csv"
    df_ebs = load_ebs_data(csv_path)

    # 2) 生成回归数据集
    lookback = 60
    X, y = create_dataset(df_ebs, lookback=lookback)

    # （可选）对 X、y 做归一化/标准化，这里仅做简单示例
    # from sklearn.preprocessing import MinMaxScaler
    # scaler_X = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    # X_reshaped = X.reshape(-1, X.shape[-1])    # (样本数 * lookback, feature_dim)
    # X_scaled = scaler_X.fit_transform(X_reshaped)
    # X = X_scaled.reshape(X.shape[0], lookback, X.shape[-1])
    # y = scaler_y.fit_transform(y.reshape(-1,1))

    # 3) 划分训练/测试集 (80% 训练, 20% 测试)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4) 构建回归模型
    model = build_lstm_model(input_shape=(lookback, 5))

    # 5) 训练模型
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 6) 评估
    #    由于是回归任务，所以返回的 metrics 只有 loss（mse），可以自己写代码计算 MAE 等
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE={test_loss:.4f}")

    # 如果做了归一化，这里需要 inverse_transform 恢复预测值
    # y_pred_scaled = model.predict(X_test)
    # y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 7) 保存模型
    os.makedirs("../models", exist_ok=True)
    save_model(model, "../models/lstm_model.h5")
    print("LSTM回归模型已保存到 ../models/lstm_model.h5")
