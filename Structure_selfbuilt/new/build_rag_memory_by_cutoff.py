import os
import sys
import subprocess
import argparse
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cut_embeddings_by_date(input_csv: str, output_csv: str, cutoff_date: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "trade_date" not in df.columns:
        raise ValueError(f"输入文件缺少 trade_date 列: {input_csv}")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).copy()

    cutoff_ts = pd.to_datetime(cutoff_date)
    df_cut = df[df["trade_date"] <= cutoff_ts].copy()
    df_cut = df_cut.sort_values(["trade_date"]).reset_index(drop=True)

    if len(df_cut) == 0:
        raise ValueError(f"裁剪后没有数据，请检查 cutoff_date={cutoff_date}")

    df_cut.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[OK] cut embeddings saved:")
    print(output_csv)
    print(f"[INFO] rows: {len(df_cut)}")
    print(f"[INFO] trade_date min: {df_cut['trade_date'].min()}")
    print(f"[INFO] trade_date max: {df_cut['trade_date'].max()}")

    return df_cut


def build_faiss_memory(
    rag_script: str,
    sector_embeddings_csv: str,
    out_dir: str,
    cosine: bool = True,
    holding_days: int = 5,
    buy_th: float = 0.55,
    sell_th: float = 0.45,
    sl_mult: float = 2.0,
    tp_mult: float = 3.0,
    sl_floor: float = 0.02,
    tp_floor: float = 0.03,
    sl_cap: float = 0.12,
    tp_cap: float = 0.20,
) -> None:
    cmd = [
        sys.executable,
        rag_script,
        "build",
        "--sector_embeddings",
        sector_embeddings_csv,
        "--out_dir",
        out_dir,
        "--holding_days",
        str(holding_days),
        "--buy_th",
        str(buy_th),
        "--sell_th",
        str(sell_th),
        "--sl_mult",
        str(sl_mult),
        "--tp_mult",
        str(tp_mult),
        "--sl_floor",
        str(sl_floor),
        "--tp_floor",
        str(tp_floor),
        "--sl_cap",
        str(sl_cap),
        "--tp_cap",
        str(tp_cap),
    ]

    if cosine:
        cmd.append("--cosine")

    print("=" * 80)
    print("[RUN BUILD]")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("=" * 80)
    print("[OK] memory built:")
    print(out_dir)


def main():
    parser = argparse.ArgumentParser(description="按截止日期裁剪 sector_embeddings_memory，并重建 RAG memory")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\data\sector_embeddings_memory.csv",
        help="原始 sector_embeddings_memory.csv",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        required=True,
        help="截止日期，例如 2021-12-31",
    )
    parser.add_argument(
        "--cut_csv",
        type=str,
        default=None,
        help="裁剪后的 embeddings 输出路径；不填则自动生成",
    )
    parser.add_argument(
        "--rag_script",
        type=str,
        default=r"D:\Anaconda3\Quant\Stock_Quant\Structure_selfbuilt\new\rag_memory_faiss.py",
        help="rag_memory_faiss.py 路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="build 后的 memory 输出目录；不填则自动生成",
    )
    parser.add_argument("--holding_days", type=int, default=5)
    parser.add_argument("--buy_th", type=float, default=0.55)
    parser.add_argument("--sell_th", type=float, default=0.45)
    parser.add_argument("--sl_mult", type=float, default=2.0)
    parser.add_argument("--tp_mult", type=float, default=3.0)
    parser.add_argument("--sl_floor", type=float, default=0.02)
    parser.add_argument("--tp_floor", type=float, default=0.03)
    parser.add_argument("--sl_cap", type=float, default=0.12)
    parser.add_argument("--tp_cap", type=float, default=0.20)
    parser.add_argument("--no_cosine", action="store_true", help="不使用 cosine，相反使用 L2")

    args = parser.parse_args()

    input_csv = args.input_csv
    cutoff_date = args.cutoff_date
    rag_script = args.rag_script

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到 input_csv: {input_csv}")
    if not os.path.exists(rag_script):
        raise FileNotFoundError(f"找不到 rag_script: {rag_script}")

    base_dir = os.path.dirname(input_csv)
    cutoff_tag = cutoff_date.replace("-", "")

    if args.cut_csv is None:
        cut_csv = os.path.join(base_dir, f"sector_embeddings_memory_cut_{cutoff_tag}.csv")
    else:
        cut_csv = args.cut_csv

    if args.out_dir is None:
        project_root = os.path.dirname(base_dir)
        out_dir = os.path.join(project_root, "new", "memory", f"broker_cut_{cutoff_tag}")
    else:
        out_dir = args.out_dir

    ensure_dir(os.path.dirname(cut_csv) or ".")
    ensure_dir(out_dir)

    # 1) 裁剪 embeddings
    cut_embeddings_by_date(
        input_csv=input_csv,
        output_csv=cut_csv,
        cutoff_date=cutoff_date,
    )

    # 2) 重建 memory
    build_faiss_memory(
        rag_script=rag_script,
        sector_embeddings_csv=cut_csv,
        out_dir=out_dir,
        cosine=not args.no_cosine,
        holding_days=args.holding_days,
        buy_th=args.buy_th,
        sell_th=args.sell_th,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        sl_floor=args.sl_floor,
        tp_floor=args.tp_floor,
        sl_cap=args.sl_cap,
        tp_cap=args.tp_cap,
    )

    print("=" * 80)
    print("[DONE]")
    print(f"cut csv   : {cut_csv}")
    print(f"memory dir: {out_dir}")


if __name__ == "__main__":
    main()
