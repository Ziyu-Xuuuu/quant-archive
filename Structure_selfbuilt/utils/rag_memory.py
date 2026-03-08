import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise ImportError("faiss 未安装，请先 pip install faiss-cpu") from e


@dataclass
class RAGDecision:
    query_row: int
    action: str
    position: float
    p_up_fused: float
    query_y_hat: float
    stop_loss: float
    take_profit: float
    topk: int
    metric: str
    stats: Dict[str, Any]
    topk_df: pd.DataFrame
    query_meta: Dict[str, Any]
    explanation: str


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def get_qt_cols(df: pd.DataFrame) -> List[str]:
    qt_cols = [c for c in df.columns if c.startswith("qt_")]
    if len(qt_cols) == 0:
        raise ValueError("找不到 qt_* 列，请确认输入数据包含 qt_0...qt_{d-1}")
    qt_cols = sorted(qt_cols, key=lambda x: int(x.split("_")[1]))
    return qt_cols


def compute_risk_params(
    row: pd.Series,
    sl_mult: float,
    tp_mult: float,
    sl_floor: float,
    tp_floor: float,
    sl_cap: float,
    tp_cap: float,
) -> Tuple[float, float, float]:
    """
    根据波动估一个止损止盈。
    优先用 vol_20 / garch_vol / 兜底 0.02
    """
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
        mdd_vals = top_df["fwd_mdd"].astype(float).values
        avg_mdd = float(np.nanmean(mdd_vals))
        mdd_q90 = float(np.nanquantile(mdd_vals, 0.90))
    else:
        avg_mdd = mdd_q90 = float(np.nan)

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
        "avg_mdd": avg_mdd,
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
        p_up = float(alpha * query_y_hat + (1.0 - alpha) * win_rate)

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


class RAGMemory:
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
    ):
        self.memory_dir = memory_dir
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

        self.index, self.metric, self.meta, self.cfg = self._load_artifacts(memory_dir)

        if "trade_date" in self.meta.columns:
            self.meta["trade_date"] = pd.to_datetime(self.meta["trade_date"], errors="coerce")

        self.meta_qt_cols = get_qt_cols(self.meta)

    def _load_artifacts(self, memory_dir: str):
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

        with open(fmt_path, "r", encoding="utf-8") as f:
            fmt = f.read().strip()

        if fmt == "parquet":
            meta_path = os.path.join(memory_dir, "meta.parquet")
            meta = pd.read_parquet(meta_path)
        else:
            meta_path = os.path.join(memory_dir, "meta.csv")
            meta = pd.read_csv(meta_path, encoding="utf-8-sig")

        metric = cfg.get("metric", "cosine")
        return index, metric, meta, cfg

    def _prepare_query_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex) and "trade_date" not in df.columns:
            df["trade_date"] = df.index

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

        if "t_index" not in df.columns:
            df["t_index"] = np.arange(len(df), dtype=int)

        if "y_hat" not in df.columns:
            df["y_hat"] = 0.5

        qt_cols = get_qt_cols(df)
        if qt_cols != self.meta_qt_cols:
            raise ValueError(
                f"query qt 维度/列名与 memory 不一致。\n"
                f"query qt cols[:5]={qt_cols[:5]} len={len(qt_cols)}\n"
                f"memory qt cols[:5]={self.meta_qt_cols[:5]} len={len(self.meta_qt_cols)}"
            )

        return df

    def _search_candidates(self, q: np.ndarray, oversample: int) -> Tuple[List[float], List[int]]:
        D_all, I_all = self.index.search(q, oversample)
        return D_all.reshape(-1).tolist(), I_all.reshape(-1).tolist()

    def _filter_candidates(
        self,
        df: pd.DataFrame,
        q_idx: int,
        I_all: List[int],
        D_all: List[float],
        exclude_ts_code: Optional[str] = None,
        strict_past_only: bool = True,
    ) -> Tuple[List[int], List[float]]:
        filtered_I: List[int] = []
        filtered_D: List[float] = []

        q_row = df.iloc[q_idx]

        q_tidx = None
        if "t_index" in df.columns and pd.notna(q_row.get("t_index", None)):
            q_tidx = int(q_row["t_index"])

        q_date = None
        if "trade_date" in df.columns and pd.notna(q_row.get("trade_date", None)):
            q_date = pd.to_datetime(q_row["trade_date"], errors="coerce")

        for idx, score in zip(I_all, D_all):
            if idx is None or idx < 0:
                continue

            cand = self.meta.iloc[idx]

            # 1) 排除目标股票自身历史
            if exclude_ts_code is not None and "ts_code" in self.meta.columns:
                cand_code = str(cand.get("ts_code", ""))
                if cand_code == exclude_ts_code:
                    continue

            # 2) 严格只允许过去样本
            if strict_past_only and q_date is not None and "trade_date" in self.meta.columns:
                cand_date = cand.get("trade_date", None)
                if pd.notna(cand_date):
                    cand_date = pd.to_datetime(cand_date, errors="coerce")
                    if pd.isna(cand_date) or cand_date >= q_date:
                        continue

            # 3) min_gap 过滤
            if q_tidx is not None and "t_index" in self.meta.columns:
                cand_tidx = cand.get("t_index", None)
                if cand_tidx is not None and pd.notna(cand_tidx):
                    cand_tidx = int(cand_tidx)
                    if abs(cand_tidx - q_tidx) < self.min_gap:
                        continue

            filtered_I.append(int(idx))
            filtered_D.append(float(score))

            if len(filtered_I) >= self.topk:
                break

        return filtered_I, filtered_D

    def _fallback_decision(self, df: pd.DataFrame, q_idx: int) -> RAGDecision:
        """
        当过滤后没有可用 RAG 候选时，退化为仅使用 Transformer 的 y_hat 决策。
        """
        q_row = df.iloc[q_idx]

        query_y_hat = float(q_row["y_hat"]) if "y_hat" in df.columns and pd.notna(q_row["y_hat"]) else 0.5
        p_up, action, position = decision_from_stats(
            query_y_hat=query_y_hat,
            stats={"topk_win_rate": np.nan},
            alpha=self.alpha,
            buy_th=self.buy_th,
            sell_th=self.sell_th,
        )

        _, sl, tp = compute_risk_params(
            row=q_row,
            sl_mult=self.sl_mult,
            tp_mult=self.tp_mult,
            sl_floor=self.sl_floor,
            tp_floor=self.tp_floor,
            sl_cap=self.sl_cap,
            tp_cap=self.tp_cap,
        )

        query_meta = {
            k: q_row[k]
            for k in ["trade_date", "date", "t_index", "ts_code", "state", "trend_regime", "vol_regime", "close"]
            if k in df.columns
        }

        explanation = (
            f"无可用RAG候选，退化为仅使用Transformer预测: "
            f"y_hat≈{query_y_hat:.2%} => {action} (pos={position:.2f}) "
            f"SL≈{sl:.2%} TP≈{tp:.2%}"
        )

        return RAGDecision(
            query_row=q_idx,
            action=action,
            position=float(position),
            p_up_fused=float(p_up),
            query_y_hat=float(query_y_hat),
            stop_loss=float(sl),
            take_profit=float(tp),
            topk=0,
            metric=self.metric,
            stats={"topk_win_rate": np.nan},
            topk_df=pd.DataFrame(),
            query_meta=query_meta,
            explanation=explanation,
        )

    def query_one(
        self,
        df: pd.DataFrame,
        q_idx: int,
        exclude_ts_code: str = None,
        strict_past_only: bool = True,
    ) -> RAGDecision:
        """
        用 df 的第 q_idx 行作为 query。
        df 应该是目标个股自身的 embeddings dataframe。
        """
        df = self._prepare_query_df(df)

        q_idx = int(q_idx)
        q_idx = max(0, min(q_idx, len(df) - 1))

        qt_all = df[self.meta_qt_cols].values.astype("float32")
        if self.metric == "cosine":
            qt_all = l2_normalize_rows(qt_all).astype("float32")

        q = qt_all[q_idx:q_idx + 1]
        oversample = max(self.topk * 5, 200)

        D_all, I_all = self._search_candidates(q, oversample=oversample)

        filtered_I, filtered_D = self._filter_candidates(
            df=df,
            q_idx=q_idx,
            I_all=I_all,
            D_all=D_all,
            exclude_ts_code=exclude_ts_code,
            strict_past_only=strict_past_only,
        )

        # 无候选时降级为仅使用 Transformer
        if len(filtered_I) == 0:
            return self._fallback_decision(df=df, q_idx=q_idx)

        top_meta = self.meta.iloc[filtered_I].copy().reset_index(drop=True)
        top_meta.insert(0, "_score", filtered_D)
        top_meta.insert(0, "_rank", np.arange(1, len(top_meta) + 1))

        stats = aggregate_stats(top_meta)

        q_row = df.iloc[q_idx]

        query_y_hat = float(q_row["y_hat"]) if "y_hat" in df.columns and pd.notna(q_row["y_hat"]) else 0.5
        p_up, action, position = decision_from_stats(
            query_y_hat=query_y_hat,
            stats=stats,
            alpha=self.alpha,
            buy_th=self.buy_th,
            sell_th=self.sell_th,
        )

        _, sl, tp = compute_risk_params(
            row=q_row,
            sl_mult=self.sl_mult,
            tp_mult=self.tp_mult,
            sl_floor=self.sl_floor,
            tp_floor=self.tp_floor,
            sl_cap=self.sl_cap,
            tp_cap=self.tp_cap,
        )

        query_meta = {
            k: q_row[k]
            for k in ["trade_date", "date", "t_index", "ts_code", "state", "trend_regime", "vol_regime", "close"]
            if k in df.columns
        }

        explanation = (
            f"qt检索top{len(top_meta)}: "
            f"win_rate≈{stats['topk_win_rate']:.2%} | "
            f"avg_ret≈{stats['avg_fwd_ret']:.4f} | "
            f"mdd_q90≈{stats['mdd_q90']:.2%} | "
            f"tp_hit≈{stats['tp_hit_rate']:.2%} sl_hit≈{stats['sl_hit_rate']:.2%} | "
            f"y_hat≈{query_y_hat:.2%} -> 融合p_up≈{p_up:.2%} => {action} (pos={position:.2f}) "
            f"SL≈{sl:.2%} TP≈{tp:.2%}"
        )

        return RAGDecision(
            query_row=q_idx,
            action=action,
            position=float(position),
            p_up_fused=float(p_up),
            query_y_hat=float(query_y_hat),
            stop_loss=float(sl),
            take_profit=float(tp),
            topk=len(top_meta),
            metric=self.metric,
            stats=stats,
            topk_df=top_meta,
            query_meta=query_meta,
            explanation=explanation,
        )
