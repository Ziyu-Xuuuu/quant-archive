# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd, numpy as np, os, torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import numpy as np
buy_signals  = {}
sell_signals = {}
today        = {}
prev         = {}

# ------------------------------------------------------------
# 通用：窗口归一化工具
# ------------------------------------------------------------
def scale_window(arr):
    """
    对形状 (lookback, feature_dim=5) 的 ndarray 做逐列 Min-Max 缩放，
    返回缩放后的 ndarray，范围在 0-1。
    """
    xmin = arr.min(axis=0)
    xmax = arr.max(axis=0)
    return (arr - xmin) / (xmax - xmin + 1e-8)

# ------------------------------------------------------------
MODEL_DIR = r"C:/Users/user/Documents/GitHub/trader/Stock_Trade/utils/models/"

# ========== Transformer ==========
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_fc = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        # ↓  变量名保持 “transformer_encoder” 不要改
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)      # ← 这里也对应修改
        return self.output_fc(x[:, -1, :])


class TransformerModel:
    def __init__(self, model_path, lookback=60):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = SimpleTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.lookback = lookback

    def predict(self, df_window):
        if len(df_window) < self.lookback:
            return np.nan
        raw = df_window[["open","high","low","close","vol"]].iloc[-self.lookback:].values
        scaled = scale_window(raw)                                      # ← NEW
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(x)[0,0].item())     # 0-1 区间预测值

# ========== LSTM ==========
class LSTMModel:
    def __init__(self, model_path, lookback=60):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model   = load_model(model_path)
        self.lookback = lookback

    def predict(self, df_window):
        if len(df_window) < self.lookback:
            return np.nan
        raw = df_window[["open","high","low","close","vol"]].iloc[-self.lookback:].values
        scaled = scale_window(raw)                                      # ← NEW
        x = scaled[np.newaxis, ...]            # (1, L, 5)
        pred = self.model.predict(x, verbose=0)[0][0]
        return float(pred)                     # 仍在 0-1 区间

# ========== HMM ==========
import joblib

def scale_window(arr):
    xmin = arr.min(axis=0)
    xmax = arr.max(axis=0)
    return (arr - xmin) / (xmax - xmin + 1e-8)

class HMMModel:
    def __init__(self, model_path, lookback=60):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model = joblib.load(model_path)
        self.lookback = lookback

    def feature_engineering(self, df_window):
        close_ret = df_window["close"].pct_change().fillna(0).values
        high_low_range = (df_window["high"] - df_window["low"]).values
        vol_change = df_window["vol"].pct_change().fillna(0).values
        feats = np.column_stack([close_ret, high_low_range, vol_change])
        return scale_window(feats)

    def predict(self, df_window):
        if len(df_window) < self.lookback:
            return np.nan
        scaled_feats = self.feature_engineering(df_window.iloc[-self.lookback:])
        states = self.model.predict(scaled_feats)
        last_state = states[-1]
        # 收益率特征在第0个位置，取means_[last_state][0]
        return float(self.model.means_[last_state][0])  

# ------------------------------------------------------------
# 组合策略：其余代码基本不变，只是初始化去掉 scaler_path
# ------------------------------------------------------------
class MetaModelStrategy:
    def __init__(self, lookback=60, selection_mode="dynamic"):
        self.lookback, self.selection_mode = lookback, selection_mode
        self.lstm_model = LSTMModel(MODEL_DIR+"lstm_model.h5", lookback)
        self.hmm_model  = HMMModel(MODEL_DIR+"hmm_model.pkl", lookback)
        self.trans_model= TransformerModel(MODEL_DIR+"transformer_model_state_dict.pt", lookback)

    def _dynamic_select_model(self, win):
        r = win["close"].pct_change().dropna().std()
        ma5 = win["close"].rolling(5).mean().iloc[-1]
        last = win["close"].iloc[-1]
        return "Transformer" if r>0.02 and last<ma5 else "LSTM" if r<0.01 and last>ma5 else "HMM"

    def _predict_by_mode(self, window):
        m = self.selection_mode
        if m=="LSTM":         return self.lstm_model.predict(window)
        if m=="HMM":          return self.hmm_model.predict(window)
        if m=="Transformer":  return self.trans_model.predict(window)
        if m=="dynamic":      return self._predict_by_mode_dynamic(window)
        if m=="average":      return np.nanmean([self.lstm_model.predict(window),
                                                 self.hmm_model.predict(window),
                                                 self.trans_model.predict(window)])
        if m=="weighted":
            preds = {"LSTM":self.lstm_model.predict(window),
                     "HMM":self.hmm_model.predict(window),
                     "Transformer":self.trans_model.predict(window)}
            w = {"LSTM":0.3,"HMM":0.2,"Transformer":0.5}
            s = sum(p*w[k] for k,p in preds.items() if not np.isnan(p))
            d = sum(w[k] for k,p in preds.items() if not np.isnan(p))
            return s/d if d else np.nan
        raise ValueError("未知 selection_mode")

    def _predict_by_mode_dynamic(self, window):
        tag = self._dynamic_select_model(window)
        return ( self.lstm_model.predict(window) if tag=="LSTM"
                 else self.hmm_model.predict(window) if tag=="HMM"
                 else self.trans_model.predict(window) )

    def generate_signals(self, df):
        preds = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            if i < self.lookback: preds.iloc[i]=np.nan; continue
            preds.iloc[i] = self._predict_by_mode(df.iloc[i-self.lookback:i])
        return preds

def sigmoid(x, k=1, x0=0):
    return 1 / (1 + np.exp(-k*(x - x0)))

# ------------------------------------------------------------
# 掘金接口 & 回测逻辑（保持之前版本，只省去 scaler_path 初始化）
# ------------------------------------------------------------
meta_strategy, day_counter = None, 0

def init(context):
    global meta_strategy, day_counter, all_symbols
    day_counter = 0
    meta_strategy = MetaModelStrategy(lookback=60, selection_mode="Transformer")

    # 把所有标的写在一个列表里
    all_symbols = [
        "SHSE.600519","SHSE.600887","SHSE.600036","SHSE.601318",
        "SZSE.000333","SZSE.000651","SHSE.600276","SHSE.600028",
        "SZSE.002475","SHSE.600585"
    ]
    for s in all_symbols:
        subscribe(symbols=s, frequency="1d", count=200)


def on_bar(context, bars):
    global day_counter, buy_signals, sell_signals, today, prev

    day_counter += 1
    if day_counter <= 61:
        print(f"冷启动第 {day_counter} 天，禁止交易")
        return

    # 如果是第一只 bar，重置所有信号容器
    if bars[0]["symbol"] == all_symbols[0]:
        buy_signals.clear()
        sell_signals.clear()
        today.clear()
        prev.clear()

    # 累加每个 symbol 的信号
    for bar in bars:
        sym = bar["symbol"]
        df = pd.DataFrame(context.data(
            symbol=sym, frequency="1d", count=200,
            fields="open,high,low,close,volume"
        ))
        df.rename(columns={"volume":"vol"}, inplace=True)

        preds = meta_strategy.generate_signals(df)
        if preds is None or len(preds) < 2:
            print(f"{sym} 预测不足")
            continue

        t, p = preds.iloc[-1], preds.iloc[-2]
        if np.isnan(t) or np.isnan(p):
            print(f"{sym} 预测含NaN")
            continue

        today[sym], prev[sym] = t, p
        diff = t - p
        if diff > 0:
            buy_signals[sym] = diff
        elif diff < 0:
            sell_signals[sym] = -diff

    # 当所有标的都生成过信号后，再统一下单
    if len(today) == len(all_symbols):
        total_buy  = sum(buy_signals.values())
        total_sell = sum(sell_signals.values())

        # 买入按权重分仓
        if total_buy > 0:
            for sym, diff in buy_signals.items():
                weight = diff / total_buy
                order_target_percent(sym, weight, PositionSide_Long, OrderType_Market)
                print(f"买入 {sym} 权重={weight:.4f}")

        # 卖出全平仓
        if total_sell > 0:
            for sym in sell_signals:
                pos = context.account().position(sym, PositionSide_Long)
                held = pos.volume if pos else 0
                if held > 0:
                    order_target_percent(sym, 0, PositionSide_Long, OrderType_Market)
                    print(f"卖出 {sym} 原持仓={held}")
                else:
                    print(f"跳过卖出 {sym}，无持仓")


# ------------------------------------------------------------
if __name__ == "__main__":
    run(
        strategy_id="meta_model_backtest",
        filename="main.py",
        mode=MODE_BACKTEST,
        token="d989ee25206f4f4a33804ca16fd164230cefe8df",
        backtest_start_time=str(datetime.now()-timedelta(days=365))[:19],
        backtest_end_time=str(datetime.now())[:19],
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10_000_000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1,
    )
