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
from tensorflow.keras.layers import InputLayer


class CompatibleInputLayer(InputLayer):
    """兼容包含 batch_shape 配置的 InputLayer."""
    def __init__(self, *args, **kwargs):
        batch_shape = kwargs.pop("batch_shape", None)
        if batch_shape is not None and "input_shape" not in kwargs and len(batch_shape) > 1:
            kwargs["input_shape"] = tuple(batch_shape[1:])
        super().__init__(*args, **kwargs)


# 新增：兼容 DTypePolicy
from tensorflow.keras.mixed_precision import Policy as BasePolicy


class DTypePolicy(BasePolicy):
    """兼容模型里使用的 'DTypePolicy'。"""
    pass


# 禁用 TensorFlow 的非错误日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 配置日志管理
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


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
        out = self.output_fc(x_last)  # 直接回归输出（通常是 scaled / ret / logret）
        return out


class TransformerModel:
    def __init__(self, model_path="utils/models/transformer_model_state_dict.pt", lookback=60, scaler_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Transformer 模型文件不存在: {model_path}")

        self.model = SimpleTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.lookback = lookback
        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict_raw(self, df_window: pd.DataFrame) -> float:
        """
        返回模型原始输出：可能是 scaled / return / log-return
        不做价格还原。
        """
        if len(df_window) < self.lookback:
            return np.nan

        if df_window[["open", "high", "low", "close", "vol"]].isna().any().any():
            logger.warning("TransformerModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[["open", "high", "low", "close", "vol"]].iloc[-self.lookback:].values
        data_torch = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(data_torch)[0, 0].item()

        # 如果训练时对 y 做了缩放，这里反标（仍是“原始量”的反标）
        if self.scaler_y is not None:
            pred = self.scaler_y.inverse_transform([[pred]])[0][0]

        return float(pred)


# ==================== LSTM 模型 ==================== #
class LSTMModel:
    def __init__(self, model_path="utils/models/lstm_model.h5", lookback=60, scaler_path=None):
        self.device = "/GPU:0" if torch.cuda.is_available() else "/CPU:0"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM 模型文件不存在: {model_path}")

        self.model = load_model(
            model_path,
            custom_objects={
                "InputLayer": CompatibleInputLayer,
                "DTypePolicy": DTypePolicy,
            },
            compile=False,
        )
        self.lookback = lookback

        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict_raw(self, df_window: pd.DataFrame) -> float:
        """
        返回模型原始输出：可能是 scaled / return / log-return
        不做价格还原。
        """
        if len(df_window) < self.lookback:
            return np.nan

        if df_window[["open", "high", "low", "close", "vol"]].isna().any().any():
            logger.warning("LSTMModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[["open", "high", "low", "close", "vol"]].iloc[-self.lookback:].values
        data = np.expand_dims(data, axis=0)  # (1, lookback, 5)

        with tf.device(self.device):
            pred = self.model.predict(data, verbose=0)[0][0]

        if self.scaler_y is not None:
            pred = self.scaler_y.inverse_transform([[pred]])[0][0]

        return float(pred)


# ==================== HMM 模型 ==================== #
class HMMModel:
    def __init__(self, model_path="utils/models/hmm_model.pkl", lookback=60, scaler_path=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HMM 模型文件不存在: {model_path}")
        self.model = load(model_path)  # GaussianHMM
        self.lookback = lookback
        self.scaler_y = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler_y = load(scaler_path)
            logger.info(f"Loaded y-scaler from {scaler_path}")

    def predict_price(self, df_window: pd.DataFrame) -> float:
        """
        HMM：用最后状态的 means_[state][3] 近似 close（你的写法本来就是 price 级别）
        """
        if len(df_window) < self.lookback:
            return np.nan
        if df_window[["open", "high", "low", "close", "vol"]].isna().any().any():
            logger.warning("HMMModel 预测时窗口数据出现 NaN，返回 NaN")
            return np.nan

        data = df_window[["open", "high", "low", "close", "vol"]].iloc[-self.lookback:].values
        state_seq = self.model.predict(data)
        last_state = state_seq[-1]
        mean_vec = self.model.means_[last_state]
        pred = mean_vec[3]  # close 均值

        if self.scaler_y is not None:
            pred = self.scaler_y.inverse_transform([[pred]])[0][0]

        return float(pred)


# ==================== MetaModelStrategy ==================== #
class MetaModelStrategy(BaseStrategy):
    call_count = 0

    def __init__(
        self,
        lstm_path="utils/models/lstm_model.h5",
        hmm_path="utils/models/hmm_model.pkl",
        transformer_path="utils/models/transformer_model_state_dict.pt",
        error_threshold=0.05,
        submodel_threshold=0.02,
        lookback=60,
        scaler_path=None,
        selection_mode="three_models",
        pred_kind="logret",   # ✅ 新增：logret / ret / price
        clip_ret=0.30,        # ✅ 新增：避免爆炸（logret/ret裁剪幅度）
    ):
        """
        pred_kind:
          - "logret": 子模型输出 = log-return（推荐）
          - "ret":    子模型输出 = return
          - "price":  子模型输出 = price（你确认模型就是价格才用）
        """
        self.lookback = lookback
        self.lstm_model = LSTMModel(model_path=lstm_path, lookback=lookback, scaler_path=scaler_path)
        self.hmm_model = HMMModel(model_path=hmm_path, lookback=lookback, scaler_path=scaler_path)
        self.transformer_model = TransformerModel(model_path=transformer_path, lookback=lookback, scaler_path=scaler_path)

        self.error_threshold = error_threshold
        self.submodel_threshold = submodel_threshold
        self.selection_mode = selection_mode

        self.pred_kind = str(pred_kind).lower()
        self.clip_ret = float(clip_ret)

    def _select_submodel(self, df_window: pd.DataFrame) -> str:
        if len(df_window) < self.lookback:
            return "LSTM"

        returns = df_window["close"].pct_change().dropna()
        volatility = returns.std()

        ma_5 = df_window["close"].rolling(window=5).mean().iloc[-1]
        last_close = df_window["close"].iloc[-1]

        if (volatility > 0.02) and (last_close < ma_5):
            if self.selection_mode == "lstm_hmm":
                return "HMM"
            elif self.selection_mode == "lstm_transformer":
                return "Transformer"
            elif self.selection_mode == "hmm_transformer":
                return "Transformer"
            else:
                return "Transformer"
        elif (volatility < 0.01) and (last_close > ma_5):
            if self.selection_mode in ("lstm_hmm", "lstm_transformer"):
                return "LSTM"
            elif self.selection_mode == "hmm_transformer":
                return "HMM"
            else:
                return "LSTM"
        else:
            if self.selection_mode == "lstm_transformer":
                return "Transformer"
            elif self.selection_mode in ("hmm_transformer", "lstm_hmm"):
                return "HMM"
            else:
                return "HMM"

    def _to_price(self, raw_pred: float, last_close: float) -> float:
        """
        把子模型输出统一还原成“预测 close 价格”
        """
        if raw_pred is None or np.isnan(raw_pred) or np.isnan(last_close):
            return np.nan

        if self.pred_kind == "price":
            return float(raw_pred)

        # logret / ret 做裁剪，避免异常点把 exp 撑爆
        r = float(np.clip(raw_pred, -self.clip_ret, self.clip_ret))

        if self.pred_kind == "logret":
            return float(last_close * np.exp(r))
        elif self.pred_kind == "ret":
            return float(last_close * (1.0 + r))
        else:
            # 兜底：当成 logret
            return float(last_close * np.exp(r))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        MetaModelStrategy.call_count += 1
        pred_price_series = pd.Series(index=df.index, dtype=float)
        total_steps = len(df)

        for i in range(total_steps):
            if i < self.lookback:
                pred_price_series.iloc[i] = np.nan
                continue

            current_data = df.iloc[i - self.lookback:i].copy()
            last_close = float(current_data["close"].iloc[-1])

            chosen_model = self._select_submodel(current_data)

            if chosen_model == "LSTM":
                raw = self.lstm_model.predict_raw(current_data)
                predicted_price = self._to_price(raw, last_close)

            elif chosen_model == "HMM":
                # HMM 你原本就是 price 级别输出（close均值）
                predicted_price = self.hmm_model.predict_price(current_data)

            else:  # Transformer
                raw = self.transformer_model.predict_raw(current_data)
                predicted_price = self._to_price(raw, last_close)

            pred_price_series.iloc[i] = predicted_price

        # 平滑
        pred_price_series = pred_price_series.rolling(window=3, min_periods=1).mean()

        # 调试对比
        logger.debug("==== Checking real close & predicted price (tail 10) ====")
        logger.debug(f"Real Close:\n{df['close'].tail(10)}")
        logger.debug(f"Predicted Price:\n{pred_price_series.tail(10)}")

        return pred_price_series
