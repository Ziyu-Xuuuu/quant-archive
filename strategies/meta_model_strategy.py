import os
import logging
import pandas as pd
import numpy as np
from joblib import load
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from strategies.base_strategy import BaseStrategy
import tensorflow as tf  # 确保正确导入 TensorFlow

# 禁用 TensorFlow 的非错误日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 配置日志管理
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Transformer 模型
class SimpleTransformer(nn.Module):
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


class TransformerModel:
    def __init__(self, model_path='utils/models/transformer_model_state_dict.pt', lookback=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查 GPU
        model_path = 'utils/models/transformer_model_state_dict.pt'
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Transformer 模型文件不存在: {model_path}")
        self.model = SimpleTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.lookback = lookback

    def predict(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < self.lookback:
            return 0.0
        data = df_window[['open', 'high', 'low', 'close', 'vol']].iloc[-self.lookback:].values
        data_torch = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(data_torch)[0, 0].item()
        return max(0.0, min(prob, 1.0))


class LSTMModel:
    def __init__(self, model_path='utils/models/lstm_model.h5', lookback=20):
        self.device = '/GPU:0' if torch.cuda.is_available() else '/CPU:0'  # 检查 GPU
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM 模型文件不存在: {model_path}")
        self.model = load_model(model_path)
        self.lookback = lookback

    def predict(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < self.lookback:
            return 0.0
        data = df_window[['open', 'high', 'low', 'close', 'vol']].iloc[-self.lookback:].values
        data = np.expand_dims(data, axis=0)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with tf.device(self.device):  # 强制使用 GPU 或 CPU
            pred = self.model.predict(data_tensor.numpy(), verbose=0)
        prob = max(0.0, min(float(pred[0][0]), 1.0))
        return prob


class HMMModel:
    def __init__(self, model_path='utils/models/hmm_model.pkl', lookback=20):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HMM 模型文件不存在: {model_path}")
        self.model = load(model_path)
        self.lookback = lookback

    def predict(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < self.lookback:
            return 0.0
        data = df_window[['open', 'high', 'low', 'close', 'vol']].iloc[-self.lookback:].values
        prob = self.model.predict_proba(data)[-1].max()
        return max(0.0, min(prob, 1.0))


class MetaModelStrategy(BaseStrategy):
    call_count = 0  # 静态变量，用于统计调用次数

    def __init__(
        self,
        lstm_path='utils/models/lstm_model.h5',
        hmm_path='utils/models/hmm_model.pkl',
        transformer_path='utils/models/transformer_model_state_dict.pt',
        error_threshold=0.05,
        submodel_threshold=0.02
    ):
        self.lstm_model = LSTMModel(model_path=lstm_path)
        self.hmm_model = HMMModel(model_path=hmm_path)
        self.transformer_model = TransformerModel(model_path=transformer_path)
        self.error_threshold = error_threshold
        self.submodel_threshold = submodel_threshold

    def _select_submodel(self, df_window: pd.DataFrame) -> str:
        recent = df_window.tail(5)
        if len(recent) < 5:
            return "LSTM"
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()
        avg_return = returns.mean()
        if volatility > 0.02:
            return "Transformer"
        elif abs(avg_return) < 0.001:
            return "LSTM"
        else:
            return "HMM"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        MetaModelStrategy.call_count += 1
        signal_series = pd.Series(index=df.index, dtype=float)

        total_steps = len(df)

        for i in range(total_steps):
            current_data = df.iloc[:i]
            if len(current_data) < 5:
                signal_series.iloc[i] = 0.0
                continue
            chosen_model = self._select_submodel(current_data)
            if chosen_model == "LSTM":
                prob = self.lstm_model.predict(current_data)
            elif chosen_model == "HMM":
                prob = self.hmm_model.predict(current_data)
            else:
                prob = self.transformer_model.predict(current_data)

            signal_series.iloc[i] = prob

        return signal_series
