"""
train_xgb_ebs.py

使用 XGBoost 训练股票预测模型，输入为5维特征 [open, high, low, close, vol]。
支持四种输出类型：
    - future_trend：预测下一天涨跌（分类）
    - future_close：预测N天后收盘价（回归）
    - future_return：预测N天收益率（回归）
    - future_volatility：预测未来是否大幅波动（多分类）
最终模型保存在 models/xgb_model_<output_type>.pkl
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# ========== 参数设置 ==========
csv_path = "C:\\Users\\user\\Documents\\GitHub\\trader\\Stock_Trade\\data\\601788_SH.csv"
output_type = "future_trend"  # 可选值: 'future_trend', 'future_close', 'future_return', 'future_volatility'
N = 3  # 未来N天（适用于收盘价、收益率、波动预测）

# ========== 数据准备 ==========
def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df

def create_label(df, output_type):
    if output_type == 'future_trend':
        df['future_close'] = df['close'].shift(-1)
        df['label'] = (df['future_close'] > df['close']).astype(int)

    elif output_type == 'future_close':
        df['label'] = df['close'].shift(-N)

    elif output_type == 'future_return':
        df['label'] = (df['close'].shift(-N) - df['close']) / df['close']

    elif output_type == 'future_volatility':
        future_range = df['high'].shift(-N) - df['low'].shift(-N)
        volatility = future_range / df['close']
        df['label'] = pd.cut(volatility, bins=[-np.inf, 0.02, 0.05, np.inf], labels=[0, 1, 2])

    else:
        raise ValueError("Invalid output_type selected!")

    df.dropna(inplace=True)
    return df

# ========== 主流程 ==========
if __name__ == "__main__":
    df = load_data(csv_path)
    df = create_label(df, output_type)

    X = df[['open', 'high', 'low', 'close', 'vol']].values
    y = df['label'].values

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 分类 or 回归模型选择
    if output_type in ['future_trend', 'future_volatility']:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                 random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ========== 模型评估 ==========
    if output_type in ['future_trend', 'future_volatility']:
        acc = accuracy_score(y_test, y_pred)
        print(f"[{output_type}] Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"[{output_type}] RMSE: {rmse:.4f}")

    # ========== 保存模型 ==========
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, f"../models/xgb_model_{output_type}.pkl")
    print(f"Model saved to ../models/xgb_model_{output_type}.pkl")
