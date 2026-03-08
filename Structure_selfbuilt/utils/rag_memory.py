# utils/rag_memory.py
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise ImportError("faiss 未安装，请先 pip install faiss-cpu") from e


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def get_qt_cols(df: pd.DataFrame) -> List[str]:
    qt_cols = [c for c in df.columns if c.startswith("qt_")]
    if len(qt_cols) == 0:
        raise ValueError("找不到 qt_* 列，请确认数据包含 qt_0...qt_{d-1}")
    qt_cols = sorted(qt_cols, key=lambda x: int(x.split("_")[1]))
    return qt_cols


def load_artifacts(memory_dir: str):
    idx_path = os.path.join(memory_dir, "faiss.index")
    cfg_path = os.path.join(memory_dir, "config.json")
    fmt_path = os.path.join(memory_dir, "meta.format")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"faiss.index not found: {idx_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    if not os.path.exists(fmt_path):
        raise FileNotFoundError(f"meta.format not found: {fmt_path}")

    index = faiss.read_index(idx_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    metric = cfg.get("metric", "cosine")

    with open(fmt_path, "r", encoding="utf-8") as f:
        fmt = f.read().strip()

    if fmt == "parquet":
        meta_path = os.path.join(memory_dir, "meta.parquet")
        meta = pd.read_parquet(meta_path)
    else:
        meta_path = os.path.join(memory_dir, "meta.csv")
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")

    return index, metric, meta


def aggregate_stats(top_df: pd.DataFrame) -> Dict[str, Any]:
    if "y" in top_df.columns:
        win_rate = float(top_df["y"].astype(float).mean())
    else:
        win_rate = float(np.nan)

    if "fwd_ret" in top_df.columns:
        rets = top_df["fwd_ret"].astype(float).values
        avg_ret = float(np.nanmean(rets))
        q25 = float(np.nanquantile(rets, 0.25))
        q50 = float(np.nanquantile(rets, 0.50))
        q75 = float(np.nanquantile(rets, 0.75))
    else:
        avg_ret = q25 = q50 = q75 = float(np.nan)

    if "fwd_mdd" in top_df.columns:
        mdd_avg = float(np.nanmean(top_df["fwd_mdd"].astype(float).values))
        mdd_q90 = float(np.nanquantile(top_df["fwd_mdd"].astype(float).values, 0.90))
    else:
        mdd_avg = mdd_q90 = float(np.nan)

    tp_hit_rate = float(top_df["tp_hit"].astype(float).mean()) if "tp_hit" in top_df.columns else float(np.nan)
    sl_hit_rate = float(top_df["sl_hit"].astype(float).mean()) if "sl_hit" in top_df.columns else float(np.nan)

    if "first_hit" in top_df.columns:
        dist = top_df["first_hit"].value_counts(normalize=True).to_dict()
        dist = {str(k): float(v) for k, v in dist.items()}
    else:
        dist = {}

    return {
        "topk_win_rate": win_rate,
        "avg_fwd_ret": avg_ret,
        "ret_q25": q25,
        "ret_q50": q50,
        "ret_q75": q75,
        "avg_mdd": mdd_avg,
        "mdd_q90": mdd_q90,
        "tp_hit_rate": tp_hit_rate,
        "sl_hit_rate": sl_hit_rate,
        "first_hit_dist": dist,
    }


def decision_from_stats(
    query_y_hat: float,
    stats: Dict[str, Any],
    alpha: float,
    buy_th: float,
    sell_th: float,
) -> Tuple[float, str, float]:
    win_rate = stats.get("topk_win_rate", np.nan)
    if np.isnan(win_rate):
        p_up = float(query_y_hat)
    else:
        p_up = float(alpha * query_y_hat + (1 - alpha) * win_rate)

    if p_up >= buy_th:
        action = "BUY"
        position = float(np.clip((p_up - 0.5) / 0.5, 0.0, 1.0))
    elif p_up <= sell_th:
        action = "SELL"
        position = 0.0
    else:
        action = "HOLD"
        position = 0.5

    return p_up, action, position


def compute_risk_params(
    row: pd.Series,
    sl_mult: float,
    tp_mult: float,
    sl_floor: float,
    tp_floor: float,
    sl_cap: float,
    tp_cap: float
) -> Tuple[float, float, float]:
    vol = None
    for c in ["vol_20", "garch_vol"]:
        if c in row.index and pd.notna(row[c]):
            vol = float(row[c])
            break
    if vol is None:
        vol = 0.02

    stop_loss = float(np.clip(vol * sl_mult, sl_floor, sl_cap))
    take_profit = float(np.clip(vol * tp_mult, tp_floor, tp_cap))
    return vol, stop_loss, take_profit


@dataclass
class RAGDecision:
    p_up: float
    action: str
    position: float
    stop_loss: float
    take_profit: float
    top_meta: pd.DataFrame
    stats: Dict[str, Any]


class RAGMemory:
    """
    FAISS 经验库检索器（回测中常驻内存）
    ✅ 严格防泄漏：cand_tidx <= q_tidx - min_gap
    ✅ 加 warmup：前 min_gap 根 bar 不检索，直接返回默认决策
    """

    def __init__(
        self,
        memory_dir: str,
        topk: int = 30,
        min_gap: int = 10,
        alpha: float = 0.5,
        buy_th: float = 0.55,
        sell_th: float = 0.45,
        sl_mult: float = 2.0,
        tp_mult: float = 3.0,
        sl_floor: float = 0.02,
        tp_floor: float = 0.03,
        sl_cap: float = 0.12,
        tp_cap: float = 0.20,
        strict_past_only: bool = True,
        warmup_action: str = "HOLD",   # "HOLD" or "SELL"
        warmup_position: float = 0.5,  # HOLD=0.5; 如果你想空仓就设 0.0
    ):
        self.index, self.metric, self.meta = load_artifacts(memory_dir)

        self.topk = int(topk)
        self.min_gap = int(min_gap)

        self.alpha = float(alpha)
        self.buy_th = float(buy_th)
        self.sell_th = float(sell_th)

        self.sl_mult = float(sl_mult)
        self.tp_mult = float(tp_mult)
        self.sl_floor = float(sl_floor)
        self.tp_floor = float(tp_floor)
        self.sl_cap = float(sl_cap)
        self.tp_cap = float(tp_cap)

        self.strict_past_only = bool(strict_past_only)
        self.warmup_action = str(warmup_action)
        self.warmup_position = float(warmup_position)

        self._printed_meta_range = False

        if "t_index" not in self.meta.columns and self.strict_past_only:
            print("[WARN][RAGMemory] meta 中缺少 t_index，strict_past_only=True 将无法严格防泄漏。")

    def _get_query_tidx(self, df: pd.DataFrame, q_idx: int) -> Optional[int]:
        if "t_index" not in df.columns:
            return None
        v = df.iloc[q_idx].get("t_index", None)
        if v is None or pd.isna(v):
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _get_cand_tidx(self, idx: int) -> Optional[int]:
        if "t_index" not in self.meta.columns:
            return None
        v = self.meta.iloc[idx].get("t_index", None)
        if v is None or pd.isna(v):
            return None
        try:
            return int(v)
        except Exception:
            return None

    def query_one(self, df: pd.DataFrame, q_idx: int) -> RAGDecision:
        qt_cols = get_qt_cols(df)
        q_idx = max(0, min(int(q_idx), len(df) - 1))

        q_tidx = self._get_query_tidx(df, q_idx)

        # DEBUG：只打印一次 meta t_index 范围（避免你一直看不到 q_idx==50）
        if not self._printed_meta_range:
            meta_min = self.meta["t_index"].min() if "t_index" in self.meta.columns else None
            meta_max = self.meta["t_index"].max() if "t_index" in self.meta.columns else None
            print("[DEBUG][RAG] meta t_index min/max =", meta_min, meta_max)
            self._printed_meta_range = True

        # ✅ warmup：严格模式下，前 min_gap 天必然无历史可用，直接返回默认
        if self.strict_past_only and q_tidx is not None and q_tidx < self.min_gap:
            _, sl, tp = compute_risk_params(
                row=df.iloc[q_idx],
                sl_mult=self.sl_mult,
                tp_mult=self.tp_mult,
                sl_floor=self.sl_floor,
                tp_floor=self.tp_floor,
                sl_cap=self.sl_cap,
                tp_cap=self.tp_cap,
            )
            empty_top = self.meta.iloc[0:0].copy()
            empty_stats = {
                "topk_win_rate": float(np.nan),
                "avg_fwd_ret": float(np.nan),
                "ret_q25": float(np.nan),
                "ret_q50": float(np.nan),
                "ret_q75": float(np.nan),
                "avg_mdd": float(np.nan),
                "mdd_q90": float(np.nan),
                "tp_hit_rate": float(np.nan),
                "sl_hit_rate": float(np.nan),
                "first_hit_dist": {},
            }
            return RAGDecision(
                p_up=float(df.iloc[q_idx].get("y_hat", 0.5)),
                action=self.warmup_action,
                position=self.warmup_position,
                stop_loss=float(sl),
                take_profit=float(tp),
                top_meta=empty_top,
                stats=empty_stats,
            )

        # query vector (1, d)
        q_vec = df.loc[df.index[q_idx], qt_cols].astype("float32").values.reshape(1, -1)
        if self.metric == "cosine":
            q_vec = l2_normalize_rows(q_vec).astype("float32")

        oversample = max(self.topk * 5, 2000)
        D_all, I_all = self.index.search(q_vec, oversample)

        I_all = I_all.reshape(-1).tolist()
        D_all = D_all.reshape(-1).tolist()

        # 额外 debug：你关心的那一行
        if q_idx == 50:
            meta_min = self.meta["t_index"].min() if "t_index" in self.meta.columns else None
            meta_max = self.meta["t_index"].max() if "t_index" in self.meta.columns else None
            print("[DEBUG][RAG] q_idx=50 q_tidx=", q_tidx, "meta t_index min/max=", meta_min, meta_max)

        filtered_I: List[int] = []
        filtered_D: List[float] = []

        for idx, score in zip(I_all, D_all):
            if idx is None:
                continue
            idx = int(idx)

            if q_tidx is not None:
                cand_tidx = self._get_cand_tidx(idx)
                if cand_tidx is not None:
                    if self.strict_past_only:
                        if cand_tidx > (q_tidx - self.min_gap):
                            continue
                    else:
                        if abs(cand_tidx - q_tidx) < self.min_gap:
                            continue

            filtered_I.append(idx)
            filtered_D.append(float(score))
            if len(filtered_I) >= self.topk:
                break

        if len(filtered_I) == 0:
            meta_min = self.meta["t_index"].min() if "t_index" in self.meta.columns else None
            meta_max = self.meta["t_index"].max() if "t_index" in self.meta.columns else None
            raise RuntimeError(
                "过滤后没有可用检索结果：\n"
                f"- q_idx={q_idx} q_tidx={q_tidx} min_gap={self.min_gap}\n"
                f"- meta t_index min/max={meta_min}/{meta_max}\n"
                "排查顺序：\n"
                "1) 确认 df 与 meta 都有 t_index，且处于同一坐标系（建议按 trade_date 排序后的 0..N-1）；\n"
                "2) min_gap 先调小测试（比如 1/2）；\n"
                "3) 经验库样本量是否足够（meta 行数太少也会筛空）。"
            )

        top_meta = self.meta.iloc[filtered_I].copy().reset_index(drop=True)
        top_meta.insert(0, "_score", filtered_D)
        top_meta.insert(0, "_rank", np.arange(1, len(top_meta) + 1))

        stats = aggregate_stats(top_meta)

        query_y_hat = float(df.iloc[q_idx].get("y_hat", 0.5))
        p_up, action, position = decision_from_stats(
            query_y_hat=query_y_hat,
            stats=stats,
            alpha=self.alpha,
            buy_th=self.buy_th,
            sell_th=self.sell_th,
        )

        _, sl, tp = compute_risk_params(
            row=df.iloc[q_idx],
            sl_mult=self.sl_mult,
            tp_mult=self.tp_mult,
            sl_floor=self.sl_floor,
            tp_floor=self.tp_floor,
            sl_cap=self.sl_cap,
            tp_cap=self.tp_cap,
        )

        return RAGDecision(
            p_up=p_up,
            action=action,
            position=position,
            stop_loss=sl,
            take_profit=tp,
            top_meta=top_meta,
            stats=stats,
        )
