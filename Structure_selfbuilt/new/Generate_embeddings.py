# Generate_embeddings.py
# 作用：
# 1) 读取 market_states_xxx_with_labels.csv
# 2) 构造滑动窗口序列 X_seq: (samples, N, feat_dim)
# 3) 加载已训练的 Keras export_model（含 ht/qt/y_hat）
# 4) 生成 ht / qt / y_hat
# 5) 保存 embeddings_ht_qt.csv
#
# 适配：
# - Train_transformer.py输出的 models/transformer_model.keras
# - 模型含自定义层 L2Normalize


import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


DEFAULT_FEATURE_COLS = ["hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_p4"]
DATE_COL_CANDIDATES = ["trade_date", "date", "交易日期"]
LABEL_COL_CANDIDATES = ["y"]


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


def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def ensure_datetime_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
    return df


def build_seq_dataset(
    df: pd.DataFrame,
    window: int,
    feature_cols: List[str],
    label_col: str,
    date_col: Optional[str] = None,
    keep_only_labeled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成序列样本：
      样本 t 使用窗口 [t-window+1, t] 的特征作为输入；
      标签使用 y[t]。
    返回：
      X_seq: (samples, window, feat_dim)
      y:     (samples,)
      end_dates: (samples,)  窗口末端日期（或索引）
      end_indices: (samples,) 窗口末端在 df 中的行号
    """
    if keep_only_labeled:
        df = df.copy()
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # 日期列
    if date_col is not None and date_col in df.columns:
        end_dates_src = df[date_col].values
    else:
        end_dates_src = np.arange(len(df))

    feat = df[feature_cols].values.astype(np.float32)
    yv = df[label_col].values

    X_list, y_list, d_list, idx_list = [], [], [], []

    for t in range(window - 1, len(df)):
        x_win = feat[t - (window - 1) : t + 1, :]  # (window, feat_dim)
        y_t = yv[t]
        if np.isnan(x_win).any() or pd.isna(y_t):
            continue
        X_list.append(x_win)
        y_list.append(int(y_t))
        d_list.append(end_dates_src[t])
        idx_list.append(t)

    if len(X_list) == 0:
        raise ValueError("没有生成任何序列样本：请检查 window、标签列 y、以及特征列是否存在且非空。")

    X_seq = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    end_dates = np.array(d_list)
    end_indices = np.array(idx_list, dtype=np.int32)
    return X_seq, y, end_dates, end_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\market_states_601788_SH_with_labels.csv")
    parser.add_argument("--model", default=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\new\transformer_model.keras")
    parser.add_argument("--output", default=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\embeddings_ht_qt.csv")
    parser.add_argument("--window", type=int, default=30, help="序列窗口长度 N（与训练一致，默认30）")
    parser.add_argument("--feature_cols", type=str, default=",".join(DEFAULT_FEATURE_COLS),
                        help="特征列名，逗号分隔。默认 hmm_p0..hmm_p4")
    parser.add_argument("--label_col", type=str, default="", help="标签列名，默认自动找 y")
    parser.add_argument("--date_col", type=str, default="", help="日期列名，默认自动找 trade_date/date/交易日期")
    parser.add_argument("--ht_layer", type=str, default="ht", help="ht 层名（默认 ht）")
    parser.add_argument("--qt_layer", type=str, default="qt", help="qt 层名（默认 qt）")
    parser.add_argument("--yhat_layer", type=str, default="y_hat", help="y_hat 层名（默认 y_hat）")
    parser.add_argument("--batch_size", type=int, default=256, help="predict 批大小")
    args = parser.parse_args()

    in_path = args.input
    model_path = args.model
    out_path = args.output

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"input not found: {in_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    if not feature_cols:
        raise ValueError("feature_cols 为空")

    # 1) 读取数据
    df = pd.read_csv(in_path, encoding="utf-8-sig")

    # date_col / label_col 自动推断
    date_col = args.date_col.strip() or pick_first_existing(df.columns.tolist(), DATE_COL_CANDIDATES)
    label_col = args.label_col.strip() or pick_first_existing(df.columns.tolist(), LABEL_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"找不到标签列。候选：{LABEL_COL_CANDIDATES}；当前列：{df.columns.tolist()}")

    # 日期排序（若有）
    if date_col is not None:
        df = ensure_datetime_column(df, date_col)

    # 2) 检查列
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列：{missing}。当前列：{df.columns.tolist()}")

    # 3) 构造序列数据
    X_seq, y, end_dates, end_idx = build_seq_dataset(
        df=df,
        window=args.window,
        feature_cols=feature_cols,
        label_col=label_col,
        date_col=date_col,
        keep_only_labeled=True,
    )
    print(f"[INFO] X_seq shape: {X_seq.shape}  (samples, window, feat_dim)")
    print(f"[INFO] y shape: {y.shape}")

    # 4) 加载模型（带自定义层）
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"L2Normalize": L2Normalize},
    )
    print("[INFO] model.input_shape:", model.input_shape)

    # 5) 构造一个导出模型：输出 ht/qt/y_hat
    ht_layer_name = args.ht_layer.strip()
    qt_layer_name = args.qt_layer.strip()
    yhat_layer_name = args.yhat_layer.strip()

    try:
        ht_tensor = model.get_layer(ht_layer_name).output
    except Exception as e:
        raise ValueError(f"找不到 ht 层 '{ht_layer_name}'，请确认训练脚本里 name='ht'。") from e

    try:
        qt_tensor = model.get_layer(qt_layer_name).output
    except Exception as e:
        raise ValueError(f"找不到 qt 层 '{qt_layer_name}'，请确认训练脚本里 name='qt'。") from e

    try:
        yhat_tensor = model.get_layer(yhat_layer_name).output
    except Exception:
        # 兜底：如果找不到 y_hat 层，就用 model.output
        yhat_tensor = model.output

    export_model = tf.keras.Model(
        inputs=model.input,
        outputs=[ht_tensor, qt_tensor, yhat_tensor],
        name="export_model_from_saved",
    )

    # 6) 生成 ht/qt/y_hat
    ht_all, qt_all, yhat_all = export_model.predict(X_seq, batch_size=args.batch_size, verbose=1)
    yhat_all = np.asarray(yhat_all).reshape(-1)
    ht_all = np.asarray(ht_all)
    qt_all = np.asarray(qt_all)

    print(f"[INFO] ht shape: {ht_all.shape}")
    print(f"[INFO] qt shape: {qt_all.shape}")
    print(f"[INFO] y_hat shape: {yhat_all.shape}")

    # 7) 输出 dataframe（附带元信息列，如果存在就带上）
    out = pd.DataFrame({
        "y": y,
        "y_hat": yhat_all,
    })

    if date_col is not None and date_col in df.columns:
        out[date_col] = end_dates
    else:
        out["t_index"] = end_dates

    # 可选 meta 列：若存在就拼上
    meta_cols = [c for c in ["ts_code", "close", "open", "high", "low", "vol",
                             "trend_regime", "vol_regime", "state",
                             "hmm_state", "hmm_state_label"]
                 if c in df.columns]
    if meta_cols:
        meta_df = df.loc[end_idx, meta_cols].reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), meta_df], axis=1)

    # ht 列
    for i in range(ht_all.shape[1]):
        out[f"ht_{i}"] = ht_all[:, i]

    # qt 列
    for i in range(qt_all.shape[1]):
        out[f"qt_{i}"] = qt_all[:, i]

    # 排序
    if date_col is not None and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col).reset_index(drop=True)

    # 8) 保存
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved embeddings to: {out_path}")
    print(out.tail())


if __name__ == "__main__":
    main()
