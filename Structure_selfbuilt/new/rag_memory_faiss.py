import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise ImportError("faiss 未安装，请先 pip install faiss-cpu") from e


DEFAULT_QUERY_CONFIG = {
    "memory_dir": r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\new\memory\broker",
    "query_embeddings": r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\embeddings_ht_qt.csv",
    "topk": 30,
    "holding_days": 5,
    "min_gap": 10
}


# =========================
# Utils
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_py(v):
    """把 numpy/pandas 标量变成 JSON 可序列化的 Python 标量"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def safe_json_dump(obj: Any, fp: str) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_py)


def detect_parquet_engine() -> Optional[str]:
    try:
        import pyarrow  # noqa
        return "pyarrow"
    except Exception:
        pass
    try:
        import fastparquet  # noqa
        return "fastparquet"
    except Exception:
        pass
    return None


def read_embeddings_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def get_qt_cols(df: pd.DataFrame) -> List[str]:
    qt_cols = [c for c in df.columns if c.startswith("qt_")]
    if len(qt_cols) == 0:
        raise ValueError("找不到 qt_* 列，请确认输入文件是否包含 qt_0...qt_{d-1}")
    qt_cols = sorted(qt_cols, key=lambda x: int(x.split("_")[1]))
    return qt_cols


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


# =========================
# Experience
# =========================
@dataclass
class Outcome:
    fwd_ret: float
    fwd_mdd: float
    tp_hit: int
    sl_hit: int
    first_hit: str   # "TP" / "SL" / "NONE"
    hit_day: int     # 第几天触发（1..H），无则 -1


def compute_outcome_long(
    df: pd.DataFrame,
    i: int,
    H: int,
    stop_loss: float,
    take_profit: float
) -> Outcome:
    """
    对第 i 行作为入场点（long），用未来 H 天的 high/low 判断 TP/SL 命中与先后。
    若没有 high/low，则退化用 close 近似。
    注意：这里 df 必须是“单股票内部按时间排序后的子表”。
    """
    entry = float(df.loc[i, "close"])
    end_i = min(i + H, len(df) - 1)

    has_high = "high" in df.columns
    has_low = "low" in df.columns

    future = df.iloc[i + 1:end_i + 1].copy()
    if future.empty:
        return Outcome(0.0, 0.0, 0, 0, "NONE", -1)

    exit_price = float(df.loc[end_i, "close"])
    fwd_ret = float(np.log(exit_price) - np.log(entry))

    if has_low:
        min_low = float(future["low"].min())
    else:
        min_low = float(future["close"].min())
    fwd_mdd = float(max(0.0, (entry - min_low) / entry))

    tp_level = entry * (1.0 + take_profit)
    sl_level = entry * (1.0 - stop_loss)

    first_hit = "NONE"
    hit_day = -1
    tp_hit = 0
    sl_hit = 0

    for k, (_, row) in enumerate(future.iterrows(), start=1):
        hi = float(row["high"]) if has_high else float(row["close"])
        lo = float(row["low"]) if has_low else float(row["close"])

        tp_now = hi >= tp_level
        sl_now = lo <= sl_level

        if tp_now and sl_now:
            first_hit = "SL"   # 保守处理
            sl_hit = 1
            tp_hit = 1
            hit_day = k
            break
        if tp_now:
            first_hit = "TP"
            tp_hit = 1
            hit_day = k
            break
        if sl_now:
            first_hit = "SL"
            sl_hit = 1
            hit_day = k
            break

    return Outcome(
        fwd_ret=fwd_ret,
        fwd_mdd=fwd_mdd,
        tp_hit=tp_hit,
        sl_hit=sl_hit,
        first_hit=first_hit,
        hit_day=hit_day
    )


def simple_policy_action(y_hat: float, buy_th: float, sell_th: float) -> str:
    if y_hat >= buy_th:
        return "BUY"
    if y_hat <= sell_th:
        return "SELL"
    return "HOLD"


def simple_policy_position(action: str, y_hat: float) -> float:
    if action == "BUY":
        pos = (y_hat - 0.5) / 0.5
        return float(np.clip(pos, 0.0, 1.0))
    if action == "SELL":
        return 0.0
    return 0.5


def compute_risk_params(
    row: pd.Series,
    sl_mult: float,
    tp_mult: float,
    sl_floor: float,
    tp_floor: float,
    sl_cap: float,
    tp_cap: float
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


def build_explanation(row: pd.Series, outcome: Outcome) -> str:
    parts = []
    for k in ["state", "trend_regime", "vol_regime"]:
        if k in row.index:
            parts.append(f"{k}={row[k]}")
    if "y_hat" in row.index and pd.notna(row["y_hat"]):
        parts.append(f"y_hat={float(row['y_hat']):.4f}")
    if "y" in row.index and pd.notna(row["y"]):
        parts.append(f"y={int(row['y'])}")
    parts.append(f"fwd_ret={outcome.fwd_ret:.4f}")
    parts.append(f"mdd={outcome.fwd_mdd:.4f}")
    parts.append(f"first_hit={outcome.first_hit}")
    return " | ".join(parts)


# =========================
# FAISS IO
# =========================
def save_artifacts(index, metric: str, df_meta: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    cfg = {"metric": metric}
    safe_json_dump(cfg, os.path.join(out_dir, "config.json"))

    engine = detect_parquet_engine()
    if engine is not None:
        df_meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False, engine=engine)
        with open(os.path.join(out_dir, "meta.format"), "w", encoding="utf-8") as f:
            f.write("parquet")
    else:
        df_meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(out_dir, "meta.format"), "w", encoding="utf-8") as f:
            f.write("csv")


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

    with open(fmt_path, "r", encoding="utf-8") as f:
        fmt = f.read().strip()

    if fmt == "parquet":
        meta_path = os.path.join(memory_dir, "meta.parquet")
        meta = pd.read_parquet(meta_path)
    else:
        meta_path = os.path.join(memory_dir, "meta.csv")
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")

    return index, cfg.get("metric", "cosine"), meta, cfg


# =========================
# Build helper
# =========================
def compute_group_outcomes(meta: pd.DataFrame, args) -> pd.DataFrame:
    """
    对多股票合并表，必须按 ts_code 分组、组内按时间排序后计算 future outcome。
    否则 i+1...i+H 会串到别的股票。
    """
    meta = meta.copy()

    if "trade_date" in meta.columns:
        meta["trade_date"] = pd.to_datetime(meta["trade_date"], errors="coerce")

    if "t_index" not in meta.columns:
        meta["t_index"] = np.arange(len(meta), dtype=int)

    actions = pd.Series(index=meta.index, dtype=object)
    positions = pd.Series(index=meta.index, dtype=float)
    vols = pd.Series(index=meta.index, dtype=float)
    sls = pd.Series(index=meta.index, dtype=float)
    tps = pd.Series(index=meta.index, dtype=float)
    fwd_rets = pd.Series(index=meta.index, dtype=float)
    fwd_mdds = pd.Series(index=meta.index, dtype=float)
    tp_hits = pd.Series(index=meta.index, dtype=float)
    sl_hits = pd.Series(index=meta.index, dtype=float)
    first_hits = pd.Series(index=meta.index, dtype=object)
    hit_days = pd.Series(index=meta.index, dtype=float)
    explains = pd.Series(index=meta.index, dtype=object)

    H = int(args.holding_days)
    buy_th = float(args.buy_th)
    sell_th = float(args.sell_th)

    if "ts_code" in meta.columns:
        groups = meta.groupby("ts_code", sort=False)
    else:
        groups = [(None, meta)]

    for _, g in groups:
        sort_cols = []
        if "trade_date" in g.columns:
            sort_cols.append("trade_date")
        if "t_index" in g.columns:
            sort_cols.append("t_index")
        if not sort_cols:
            sort_cols = list(g.columns[:1])

        g = g.sort_values(sort_cols).copy().reset_index()
        # g["index"] 是原 meta 的索引

        for i in range(len(g)):
            row = g.iloc[i]
            global_idx = int(row["index"])

            y_hat = float(row["y_hat"]) if "y_hat" in g.columns and pd.notna(row["y_hat"]) else 0.5
            action = simple_policy_action(y_hat, buy_th=buy_th, sell_th=sell_th)
            position = simple_policy_position(action, y_hat)

            vol, sl, tp = compute_risk_params(
                row=row,
                sl_mult=float(args.sl_mult),
                tp_mult=float(args.tp_mult),
                sl_floor=float(args.sl_floor),
                tp_floor=float(args.tp_floor),
                sl_cap=float(args.sl_cap),
                tp_cap=float(args.tp_cap),
            )

            outcome = compute_outcome_long(g, i=i, H=H, stop_loss=sl, take_profit=tp)
            exp = build_explanation(row, outcome)

            actions.loc[global_idx] = action
            positions.loc[global_idx] = position
            vols.loc[global_idx] = vol
            sls.loc[global_idx] = sl
            tps.loc[global_idx] = tp
            fwd_rets.loc[global_idx] = outcome.fwd_ret
            fwd_mdds.loc[global_idx] = outcome.fwd_mdd
            tp_hits.loc[global_idx] = outcome.tp_hit
            sl_hits.loc[global_idx] = outcome.sl_hit
            first_hits.loc[global_idx] = outcome.first_hit
            hit_days.loc[global_idx] = outcome.hit_day
            explains.loc[global_idx] = exp

    meta["action"] = actions
    meta["position"] = positions
    meta["vol_used"] = vols
    meta["stop_loss"] = sls
    meta["take_profit"] = tps
    meta["holding_days"] = H
    meta["fwd_ret"] = fwd_rets
    meta["fwd_mdd"] = fwd_mdds
    meta["tp_hit"] = tp_hits
    meta["sl_hit"] = sl_hits
    meta["first_hit"] = first_hits
    meta["hit_day"] = hit_days
    meta["explain"] = explains

    return meta


# =========================
# Build
# =========================
def build_memory(args) -> None:
    df = read_embeddings_csv(args.sector_embeddings)
    qt_cols = get_qt_cols(df)

    qt = df[qt_cols].values.astype("float32")

    if args.cosine:
        qt = l2_normalize_rows(qt).astype("float32")
        metric = "cosine"
        index = faiss.IndexFlatIP(qt.shape[1])
    else:
        metric = "l2"
        index = faiss.IndexFlatL2(qt.shape[1])

    index.add(qt)

    meta = df.copy()
    if "t_index" not in meta.columns:
        meta["t_index"] = np.arange(len(meta), dtype=int)

    meta = compute_group_outcomes(meta, args)

    save_artifacts(index=index, metric=metric, df_meta=meta, out_dir=args.out_dir)

    print(f"[OK] memory built: {args.out_dir}")
    print(f" - index size: {index.ntotal}")
    print(f" - metric: {metric}")
    print(f" - meta cols: {meta.shape[1]}")
    print(f" - holding_days(H): {int(args.holding_days)}")


# =========================
# Query helper
# =========================
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
        mdd_avg = float(np.nanmean(mdd_vals))
        mdd_q90 = float(np.nanquantile(mdd_vals, 0.90))
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


# =========================
# Query
# =========================
def query_memory(args) -> None:
    index, metric, meta, cfg = load_artifacts(args.memory_dir)

    # 这里读取的是“目标个股”的 embeddings，而不是板块总表
    df = read_embeddings_csv(args.query_embeddings)
    qt_cols = get_qt_cols(df)
    qt_all = df[qt_cols].values.astype("float32")

    if metric == "cosine":
        qt_all = l2_normalize_rows(qt_all).astype("float32")

    q_idx = int(args.query_row) if args.query_row is not None else (len(df) - 1)
    q_idx = max(0, min(q_idx, len(df) - 1))

    q = qt_all[q_idx:q_idx + 1]

    k = int(args.topk)
    oversample = max(k * 5, 200)

    D_all, I_all = index.search(q, oversample)
    I_all = I_all.reshape(-1).tolist()
    D_all = D_all.reshape(-1).tolist()

    q_tidx = int(df.loc[q_idx, "t_index"]) if "t_index" in df.columns and pd.notna(df.loc[q_idx, "t_index"]) else None
    min_gap = int(args.min_gap)
    exclude_ts_code = args.exclude_ts_code

    filtered_I = []
    filtered_D = []

    for idx, score in zip(I_all, D_all):
        if idx is None or idx < 0:
            continue

        # 排除目标股票自身历史
        if exclude_ts_code is not None and "ts_code" in meta.columns:
            cand_code = str(meta.iloc[idx].get("ts_code", ""))
            if cand_code == exclude_ts_code:
                continue

        # 时间间隔过滤，防止过近样本泄漏
        if q_tidx is not None and "t_index" in meta.columns:
            cand_tidx = meta.iloc[idx].get("t_index", None)
            if cand_tidx is not None and pd.notna(cand_tidx):
                cand_tidx = int(cand_tidx)
                if abs(cand_tidx - q_tidx) < min_gap:
                    continue

        filtered_I.append(int(idx))
        filtered_D.append(float(score))
        if len(filtered_I) >= k:
            break

    if len(filtered_I) == 0:
        raise RuntimeError("过滤后没有可用检索结果：请调小 --min_gap 或增大数据量。")

    top_meta = meta.iloc[filtered_I].copy().reset_index(drop=True)
    top_meta.insert(0, "_score", filtered_D)
    top_meta.insert(0, "_rank", np.arange(1, len(top_meta) + 1))

    print("\n========== QUERY ==========")
    show_cols = [
        c for c in [
            "trade_date", "date", "t_index", "ts_code",
            "state", "trend_regime", "vol_regime",
            "close", "y", "y_hat"
        ] if c in df.columns
    ]
    print(df.loc[q_idx, show_cols] if len(show_cols) > 0 else df.loc[q_idx])

    print("\n========== TOPK RESULTS (filtered) ==========")
    keep = [
        c for c in [
            "_rank", "_score", "t_index", "ts_code", "state", "trend_regime",
            "vol_regime", "close", "action", "position", "stop_loss",
            "take_profit", "holding_days", "y", "y_hat", "fwd_ret", "fwd_mdd",
            "tp_hit", "sl_hit", "first_hit", "hit_day"
        ] if c in top_meta.columns
    ]
    print(top_meta[keep].to_string(index=False))

    stats = aggregate_stats(top_meta)

    query_y_hat = float(df.loc[q_idx, "y_hat"]) if "y_hat" in df.columns and pd.notna(df.loc[q_idx, "y_hat"]) else 0.5
    p_up, action, position = decision_from_stats(
        query_y_hat=query_y_hat,
        stats=stats,
        alpha=float(args.alpha),
        buy_th=float(args.buy_th),
        sell_th=float(args.sell_th),
    )

    q_row = df.loc[q_idx]
    vol, sl, tp = compute_risk_params(
        row=q_row,
        sl_mult=float(args.sl_mult),
        tp_mult=float(args.tp_mult),
        sl_floor=float(args.sl_floor),
        tp_floor=float(args.tp_floor),
        sl_cap=float(args.sl_cap),
        tp_cap=float(args.tp_cap),
    )

    decision = {
        "query_row": q_idx,
        "query_meta": {
            k: to_py(q_row[k])
            for k in ["trade_date", "date", "t_index", "ts_code", "state", "trend_regime", "vol_regime", "close"]
            if k in df.columns
        },
        "topk": k,
        "metric": metric,
        "min_gap": min_gap,
        "exclude_ts_code": exclude_ts_code,
        "stats": {
            **{k: to_py(v) for k, v in stats.items()},
            "query_y_hat": float(query_y_hat),
            "p_up_fused": float(p_up),
        },
        "decision": {
            "action": action,
            "position": float(position),
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "holding_days": int(args.holding_days),
        },
        "explanation": (
            f"qt检索top{k}（min_gap={min_gap}）: "
            f"win_rate≈{stats['topk_win_rate']:.2%} | "
            f"avg_ret≈{stats['avg_fwd_ret']:.4f} | "
            f"mdd_q90≈{stats['mdd_q90']:.2%} | "
            f"tp_hit≈{stats['tp_hit_rate']:.2%} sl_hit≈{stats['sl_hit_rate']:.2%} | "
            f"y_hat≈{query_y_hat:.2%} -> 融合p_up≈{p_up:.2%} => {action} (pos={position:.2f}) "
            f"SL≈{sl:.2%} TP≈{tp:.2%} H={int(args.holding_days)}"
        )
    }

    print("\n========== DECISION JSON ==========")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=to_py))

    if args.out_csv:
        ensure_dir(os.path.dirname(args.out_csv) or ".")
        top_meta.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] saved topk csv: {args.out_csv}")

    if args.out_json:
        ensure_dir(os.path.dirname(args.out_json) or ".")
        safe_json_dump(decision, args.out_json)
        print(f"[OK] saved decision json: {args.out_json}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # BUILD MEMORY
    p_build = sub.add_parser("build")
    p_build.add_argument("--sector_embeddings", required=True, help="sector_embeddings_memory.csv")
    p_build.add_argument("--out_dir", required=True, help="memory 输出目录")
    p_build.add_argument("--cosine", action="store_true", help="使用 cosine 相似度")
    p_build.add_argument("--holding_days", type=int, default=5, help="forward holding days")
    p_build.add_argument("--buy_th", type=float, default=0.55, help="BUY 阈值")
    p_build.add_argument("--sell_th", type=float, default=0.45, help="SELL 阈值")
    p_build.add_argument("--sl_mult", type=float, default=2.0)
    p_build.add_argument("--tp_mult", type=float, default=3.0)
    p_build.add_argument("--sl_floor", type=float, default=0.02)
    p_build.add_argument("--tp_floor", type=float, default=0.03)
    p_build.add_argument("--sl_cap", type=float, default=0.12)
    p_build.add_argument("--tp_cap", type=float, default=0.20)
    p_build.set_defaults(func=build_memory)

    # QUERY MEMORY
    p_query = sub.add_parser("query")
    p_query.add_argument("--memory_dir", default=DEFAULT_QUERY_CONFIG["memory_dir"], help="memory 目录")
    p_query.add_argument("--query_embeddings", default=DEFAULT_QUERY_CONFIG["query_embeddings"], help="目标个股 embeddings_ht_qt.csv")
    p_query.add_argument("--query_row", type=int, default=None, help="使用第几行作为 query（默认最后一行）")
    p_query.add_argument("--topk", type=int, default=DEFAULT_QUERY_CONFIG["topk"], help="检索 topK")
    p_query.add_argument("--min_gap", type=int, default=DEFAULT_QUERY_CONFIG["min_gap"], help="最小时间间隔（防止信息泄漏）")
    p_query.add_argument("--exclude_ts_code", type=str, default=None, help="查询时排除该股票代码")

    # DECISION PARAMETERS
    p_query.add_argument("--holding_days", type=int, default=DEFAULT_QUERY_CONFIG["holding_days"], help="持有天数")
    p_query.add_argument("--alpha", type=float, default=0.5, help="模型预测 vs 经验胜率融合权重")
    p_query.add_argument("--buy_th", type=float, default=0.55, help="BUY 阈值")
    p_query.add_argument("--sell_th", type=float, default=0.45, help="SELL 阈值")

    # RISK CONTROL
    p_query.add_argument("--sl_mult", type=float, default=2.0)
    p_query.add_argument("--tp_mult", type=float, default=3.0)
    p_query.add_argument("--sl_floor", type=float, default=0.02)
    p_query.add_argument("--tp_floor", type=float, default=0.03)
    p_query.add_argument("--sl_cap", type=float, default=0.12)
    p_query.add_argument("--tp_cap", type=float, default=0.20)

    # OUTPUT
    p_query.add_argument("--out_csv", type=str, default=None, help="保存检索结果 csv")
    p_query.add_argument("--out_json", type=str, default=None, help="保存 decision json")
    p_query.set_defaults(func=query_memory)

    args = parser.parse_args()
    print("[INFO] cmd =", args.cmd)
    args.func(args)


if __name__ == "__main__":
    main()
