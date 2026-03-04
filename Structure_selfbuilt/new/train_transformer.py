# Train_transformer.py
# 作用：
# 1) 读取 market_states_601788_SH_with_labels.csv
# 2) 构造序列样本 X: (samples, 30, 5)，标签 y: (samples,)
# 3) 训练 Transformer encoder + classifier
# 4) 显式产生 ht/qt/y_hat（层名固定为 ht/qt/y_hat）
# 5) 保存 export_model 到 .keras


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

# ====== 路径（按需改）======
DATA_PATH = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\market_states_601788_SH_with_labels.csv"
MODEL_OUT = r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\transformer_model.keras"

# ====== 超参数 ======
N = 30  # 序列窗口长度
FEATURE_COLS = ["hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"]
LABEL_COL = "y"
DATE_COL = "trade_date"

D_MODEL = 64
HT_DIM = 64
QT_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.1


class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.l2_normalize(x, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


def make_seq_dataset(df: pd.DataFrame, window: int, feature_cols, label_col="y"):
    """
    每个样本 t 用过去 window 天 (t-window+1..t) 的特征作为输入；
    标签使用 y[t]。
    输出：
      X: (samples, window, feat_dim)
      y: (samples,)
    """
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X_list, y_list = [], []
    for t in range(window - 1, len(df)):
        x_win = df.loc[t - (window - 1):t, feature_cols].values  # (window, feat_dim)
        y_t = df.loc[t, label_col]
        if np.isnan(x_win).any() or pd.isna(y_t):
            continue
        X_list.append(x_win.astype(np.float32))
        y_list.append(int(y_t))

    if len(X_list) == 0:
        raise ValueError("没有生成任何序列样本。请检查 window、特征列、标签列是否有效。")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def transformer_encoder_block(x, num_heads, key_dim, dropout, name_prefix: str):
    """一个简单 Transformer Encoder block：MHA + residual + LN + FFN + residual + LN"""
    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f"{name_prefix}_mha")(x, x)
    attn = layers.Dropout(dropout, name=f"{name_prefix}_attn_drop")(attn)
    x = layers.Add(name=f"{name_prefix}_attn_res")([x, attn])
    x = layers.LayerNormalization(name=f"{name_prefix}_attn_ln")(x)

    # Feed-forward
    ffn = layers.Dense(key_dim * 2, activation="relu", name=f"{name_prefix}_ffn_dense1")(x)
    ffn = layers.Dropout(dropout, name=f"{name_prefix}_ffn_drop")(ffn)
    ffn = layers.Dense(key_dim, name=f"{name_prefix}_ffn_dense2")(ffn)

    x = layers.Add(name=f"{name_prefix}_ffn_res")([x, ffn])
    x = layers.LayerNormalization(name=f"{name_prefix}_ffn_ln")(x)
    return x


def build_transformer_seq_models(seq_len: int, feat_dim: int):
    """
    返回两个模型：
    - train_model: 输入 (seq_len, feat_dim) 输出 y_hat
    - export_model: 输入 (seq_len, feat_dim) 输出 [ht, qt, y_hat]
    """
    inp = layers.Input(shape=(seq_len, feat_dim), name="x_seq")

    # 线性投影到 d_model
    x = layers.Dense(D_MODEL, name="proj_in")(inp)

    # 两个 encoder block
    x = transformer_encoder_block(x, num_heads=NUM_HEADS, key_dim=D_MODEL, dropout=DROPOUT, name_prefix="enc1")
    x = transformer_encoder_block(x, num_heads=NUM_HEADS, key_dim=D_MODEL, dropout=DROPOUT, name_prefix="enc2")

    # Pool -> ht
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    ht = layers.Dense(HT_DIM, activation="relu", name="ht")(x)

    # qt projection + L2 normalize
    qt_raw = layers.Dense(QT_DIM, name="qt_raw")(ht)
    qt = L2Normalize(axis=-1, name="qt")(qt_raw)

    # classifier head
    x_cls = layers.Dropout(0.5, name="cls_drop")(ht)
    y_hat = layers.Dense(1, activation="sigmoid", name="y_hat")(x_cls)

    train_model = Model(inputs=inp, outputs=y_hat, name="transformer_seq_classifier")
    export_model = Model(inputs=inp, outputs=[ht, qt, y_hat], name="transformer_seq_export")
    return train_model, export_model


def main():
    # 1) 读数据
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 2) 插补缺失（只对 feature_cols）
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"缺少特征列: {c}. 当前列: {df.columns.tolist()}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"缺少标签列: {LABEL_COL}. 当前列: {df.columns.tolist()}")

    imputer = SimpleImputer(strategy="mean")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS].astype(np.float32))

    # 3) 构造序列数据 (samples, 30, 5)
    X, y = make_seq_dataset(df, window=N, feature_cols=FEATURE_COLS, label_col=LABEL_COL)
    print("[INFO] X shape:", X.shape, "y shape:", y.shape, "pos_rate:", float(y.mean()))

    # 4) 时间序列切分（不 shuffle）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # 5) 建模
    train_model, export_model = build_transformer_seq_models(seq_len=N, feat_dim=X.shape[2])
    train_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 6) 训练
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    train_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early],
        verbose=1
    )

    # 7) 评估
    loss, acc = train_model.evaluate(X_test, y_test, verbose=0)
    print(f"[TEST] loss={loss:.4f} acc={acc:.4f}")

    y_prob = train_model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # 8) 保存 export_model（包含 ht/qt/y_hat）
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    export_model.save(MODEL_OUT)
    print("[OK] Saved export model to:", MODEL_OUT)
    print("[INFO] export_model.input_shape:", export_model.input_shape)
    print("[INFO] export_model outputs:", [o.name for o in export_model.outputs])

if __name__ == "__main__":
    main()














