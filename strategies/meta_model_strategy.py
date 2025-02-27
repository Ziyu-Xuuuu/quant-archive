import os
import logging
import pandas as pd
import numpy as np
from joblib import load
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
import tensorflow as tf

from strategies.base_strategy import BaseStrategy

# 禁用 TensorFlow 的非错误日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 配置日志管理
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# ==================== Transformer 模型 ==================== #
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_fc = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x_last = x[:, -1, :]
        out = self.output_fc(x_last)  # 直接回归输出
        return out

class TransformerModel:
    def __init__(self, model_path='utils/models/transformer_model_state_dict.pt', lookback=60,
                 scaler_path=None):
        """
        如果有对 y(收盘价) 做缩放，就传入 scaler_path，
        并在预测后做 inverse_transform。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Transformer 模型文件不存在: {model_path}")
        self.model = SimpleTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.lookback = lookback

        # 如果有保存好的 y-scaler，这里加载
        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < self.lookback:
            return np.nan

        if df_window[['open', 'high', 'low', 'close', 'vol']].isna().any().any():
            logger.warning("TransformerModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[['open', 'high', 'low', 'close', 'vol']].iloc[-self.lookback:].values
        data_torch = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(data_torch)[0, 0].item()

        # 如果训练时对 y 做了缩放，就反标
        if self.scaler_y is not None:
            # scaler_y.inverse_transform 需要 2D
            pred_real = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            return float(pred_real)
        else:
            return float(pred_scaled)

# ==================== LSTM 模型 ==================== #
class LSTMModel:
    def __init__(self, model_path='utils/models/lstm_model.h5', lookback=60,
                 scaler_path=None):
        self.device = '/GPU:0' if torch.cuda.is_available() else '/CPU:0'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM 模型文件不存在: {model_path}")
        self.model = load_model(model_path)
        self.lookback = lookback

        # 如果有保存好的 y-scaler
        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < self.lookback:
            return np.nan

        if df_window[['open', 'high', 'low', 'close', 'vol']].isna().any().any():
            logger.warning("LSTMModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[['open','high','low','close','vol']].iloc[-self.lookback:].values
        data = np.expand_dims(data, axis=0)  # shape: (1, lookback, 5)

        with tf.device(self.device):
            pred_scaled = self.model.predict(data, verbose=0)[0][0]

        # 若有 scaler_y, 做反标
        if self.scaler_y is not None:
            pred_real = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            return float(pred_real)
        else:
            return float(pred_scaled)


# ==================== HMM 模型 ==================== #
class HMMModel:
    def __init__(self, model_path='utils/models/hmm_model.pkl', lookback=60,
                 scaler_path=None):
        """
        如果你想让HMM输出价格，需要有“回归HMM”的思路或把状态映射到close均值。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HMM 模型文件不存在: {model_path}")
        self.model = load(model_path)  # 默认: GaussianHMM (无监督)
        self.lookback = lookback
        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict(self, df_window: pd.DataFrame) -> float:
        """
        如果你直接用 .predict(data)，只会得到状态序列，如 [0,1,1,0...].
        这里演示一种“用最后时刻状态 -> 取close均值”来近似输出价格。
        """
        if len(df_window) < self.lookback:
            return np.nan
        if df_window[['open', 'high', 'low', 'close', 'vol']].isna().any().any():
            logger.warning("HMMModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[['open', 'high', 'low', 'close', 'vol']].iloc[-self.lookback:].values

        # 先获得最后时刻的状态
        state_seq = self.model.predict(data)  # shape=(lookback,)
        last_state = state_seq[-1]

        # 在 model.means_ 中，每个 state 有一个均值向量 (5维)
        # 其中第0维=avg open, 第1维=avg high, 第2维=avg low, 第3维=avg close, 第4维=avg vol
        # 你可选用第3维(对应 close)当“预测值”
        # 这只是一种简单映射，未必是真正的回归预测。
        mean_vec = self.model.means_[last_state]  # shape=(5,)
        pred_scaled = mean_vec[3]  # 第3个下标表示 close

        # 如果你对 close 做过缩放，这里反标
        if self.scaler_y is not None:
            pred_real = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            return float(pred_real)
        else:
            return float(pred_scaled)


# ==================== MetaModelStrategy ==================== #
class MetaModelStrategy(BaseStrategy):
    call_count = 0

    def __init__(
        self,
        lstm_path='utils/models/lstm_model.h5',
        hmm_path='utils/models/hmm_model.pkl',
        transformer_path='utils/models/transformer_model_state_dict.pt',
        error_threshold=0.05,
        submodel_threshold=0.02,
        lookback=60,  # 确保这个参数存在
        scaler_path=None
    ):
        self.lookback = lookback  # 这里确保 lookback 被正确存储
        self.lstm_model = LSTMModel(model_path=lstm_path, lookback=lookback, scaler_path=scaler_path)
        self.hmm_model = HMMModel(model_path=hmm_path, lookback=lookback, scaler_path=scaler_path)
        self.transformer_model = TransformerModel(model_path=transformer_path, lookback=lookback, scaler_path=scaler_path)
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
        pred_price_series = pd.Series(index=df.index, dtype=float)
        total_steps = len(df)

        for i in range(total_steps):
            # 若当前 i < self.lookback，就无法拿到 60 条历史
            if i < self.lookback:
                pred_price_series.iloc[i] = np.nan
                continue

            # 用最近 60 条做窗口
            # 即 [i - lookback, i)
            current_data = df.iloc[i - self.lookback:i].copy()

            # 然后根据子模型来预测
            chosen_model = self._select_submodel(current_data)
            if chosen_model == "LSTM":
                predicted_price = self.lstm_model.predict(current_data)
            elif chosen_model == "HMM":
                predicted_price = self.hmm_model.predict(current_data)
            else:  # Transformer
                predicted_price = self.transformer_model.predict(current_data)

            pred_price_series.iloc[i] = predicted_price
            # logger.debug(f"Index={i}, chosen_model={chosen_model}, predicted_price={predicted_price}")

        # 对预测结果做一个简单平滑 (可选)
        pred_price_series = pred_price_series.rolling(window=3, min_periods=1).mean()

        # ========== 调试打印对比 ==========
        logger.debug("==== Checking real close & predicted price (tail 10) ====")
        close_tail = df['close'].tail(10)
        pred_tail = pred_price_series.tail(10)
        logger.debug(f"Real Close:\n{close_tail}")
        logger.debug(f"Predicted Price:\n{pred_tail}")

        return pred_price_series