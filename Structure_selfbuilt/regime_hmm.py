import os
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# 1. 路径与列名映射
INPUT_CSV = r"C:\Users\23168\Desktop\宁德时代标点\300750_宁德时代-历史数据.csv"
OUTPUT_CSV = r"C:\Users\23168\Desktop\market_states_300750.csv"

COLUMN_MAP = {
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交数量(股)": "volume",
    "成交金额(元)": "amount",
}


# 2. 技术指标定义
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 指标（Wilder 风格，处理 avg_loss==0 的极端情况）
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    # 极端情况：avg_loss=0 且 avg_gain>0 -> RSI 接近 100
    rsi_val = np.where((avg_loss == 0) & (avg_gain > 0), 100, rsi_val)

    return pd.Series(rsi_val, index=series.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder ADX 指标：
    - ADX < 20: 无明显趋势
    - ADX > 25: 趋势明显
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_val


def rolling_linreg_tstat(series: pd.Series, window: int = 60) -> pd.Series:
    """
    对价格序列做滚动线性回归 y ~ t，返回回归斜率的 t-stat
    """
    series = series.astype(float)

    def _tstat(y: pd.Series) -> float:
        y = y.values
        n = len(y)
        if n < 3:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()

        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x == 0:
            return np.nan

        beta = cov_xy / var_x
        y_hat = y_mean + beta * (x - x_mean)
        residuals = y - y_hat
        s2 = np.sum(residuals ** 2) / (n - 2)
        se_beta = np.sqrt(s2 / var_x)
        if se_beta == 0:
            return np.nan

        t_stat = beta / se_beta
        return float(t_stat)

    return series.rolling(window).apply(_tstat, raw=False)


def hurst_exponent(series: pd.Series) -> float:
    """
    基于 R/S 分析的 Hurst 指数估计：
    - H > 0.5  趋势性
    - H ≈ 0.5 随机游走
    - H < 0.5 均值回复
    """
    y = series.dropna().values
    n = len(y)
    if n < 20:
        return np.nan

    max_k = int(np.log2(n))
    if max_k < 2:
        return np.nan

    rs_list = []
    window_sizes = []

    for k in range(2, max_k + 1):
        win = 2 ** k
        if win > n:
            break
        n_segments = n // win
        rs_vals = []
        for i in range(n_segments):
            seg = y[i * win:(i + 1) * win]
            mean = seg.mean()
            dev = seg - mean
            cum_dev = np.cumsum(dev)
            R = cum_dev.max() - cum_dev.min()
            S = seg.std()
            if S > 0:
                rs_vals.append(R / S)
        if len(rs_vals) > 0:
            rs_list.append(np.mean(rs_vals))
            window_sizes.append(win)

    if len(rs_list) < 2:
        return np.nan

    log_rs = np.log(rs_list)
    log_win = np.log(window_sizes)
    slope = np.polyfit(log_win, log_rs, 1)[0]
    return float(slope)


def rolling_hurst(series: pd.Series, window: int = 128) -> pd.Series:
    """
    滚动 Hurst 指数
    """
    return series.rolling(window).apply(hurst_exponent, raw=False)


def macd(series: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD 指标：
    - macd_line: 快 EMA - 慢 EMA
    - signal_line: macd_line 的 EMA
    - macd_hist: macd_line - signal_line
    - macd_cross_up / macd_cross_down: 金叉 / 死叉 标志
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    cross_up = (hist > 0) & (hist.shift(1) <= 0)
    cross_down = (hist < 0) & (hist.shift(1) >= 0)

    out = pd.DataFrame(
        {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "macd_hist": hist,
            "macd_cross_up": cross_up,
            "macd_cross_down": cross_down,
        }
    )
    return out


def bollinger_bands(series: pd.Series,
                    period: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """
    布林带：
    - bb_middle: 中轨
    - bb_upper, bb_lower: 上下轨
    - bb_width: (upper - lower) / middle
    """
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper = ma + num_std * std
    lower = ma - num_std * std
    bb_width = (upper - lower) / ma

    df = pd.DataFrame(
        {
            "bb_middle": ma,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": bb_width,
        }
    )
    return df


def calibrate_bb_width_thresholds(bb_width: pd.Series,
                                  q_low: float = 0.3,
                                  q_high: float = 0.7) -> tuple[float, float]:
    """
    用历史分位数来标定布林带宽阈值：
    - 低于 q_low -> 低波动收敛
    - 高于 q_high -> 高波动发散
    """
    bb_width = bb_width.dropna()
    low = bb_width.quantile(q_low)
    high = bb_width.quantile(q_high)
    return low, high


def realized_vol(returns: pd.Series, window: int = 20,
                 annualize: bool = False, trading_days: int = 252) -> pd.Series:
    """
    realized volatility（历史波动率）:
    vol_t = sqrt( mean( r_{t-window+1:t}^2 ) )
    """
    r = returns.astype(float)
    vol = r.rolling(window).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol


def upside_downside_vol(returns: pd.Series, window: int = 20,
                        annualize: bool = False, trading_days: int = 252) -> tuple[pd.Series, pd.Series]:
    """
    上行/下行波动率：
    - upside_vol: 只对 r>0 部分计算 RMS
    - downside_vol: 只对 r<0 部分计算 RMS
    """
    r = returns.astype(float)

    up = np.where(r > 0, r, 0.0)
    down = np.where(r < 0, r, 0.0)

    up = pd.Series(up, index=r.index)
    down = pd.Series(down, index=r.index)

    up_vol = up.rolling(window).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
    down_vol = down.rolling(window).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)

    if annualize:
        up_vol = up_vol * np.sqrt(trading_days)
        down_vol = down_vol * np.sqrt(trading_days)

    return up_vol, down_vol


def compute_garch_vol(returns: pd.Series) -> pd.Series:
    """
    对收益率序列拟合 GARCH(1,1)，返回 in-sample 条件波动率
    """
    r = returns.dropna()

    if len(r) < 200:
        return pd.Series(np.nan, index=returns.index)

    am = arch_model(r * 100,
                    mean="constant",
                    vol="GARCH",
                    p=1,
                    q=1,
                    dist="normal")
    res = am.fit(update_freq=0, disp="off")

    cond_vol = res.conditional_volatility / 100.0
    cond_vol = cond_vol.reindex(returns.index)
    return cond_vol


# 3. 特征计算
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 df 至少包含：['open','high','low','close']
    返回添加好所有后续需要的特征
    """
    df = df.copy()

    # log 收益
    df["ret"] = np.log(df["close"]).diff()

    # 均线（快速慢速 + 多周期）
    df["ma_fast"] = df["close"].rolling(20).mean()
    df["ma_slow"] = df["close"].rolling(60).mean()

    df["ma_short"] = df["close"].rolling(5).mean()
    df["ma_mid"] = df["close"].rolling(20).mean()
    df["ma_long"] = df["close"].rolling(60).mean()

    # RSI
    df["rsi"] = rsi(df["close"], period=14)

    # MACD
    macd_df = macd(df["close"])
    df["macd"] = macd_df["macd_line"]
    df["macd_signal"] = macd_df["signal_line"]
    df["macd_hist"] = macd_df["macd_hist"]
    df["macd_cross_up"] = macd_df["macd_cross_up"]
    df["macd_cross_down"] = macd_df["macd_cross_down"]

    # 布林带
    bb = bollinger_bands(df["close"])
    df["bb_ma"] = bb["bb_middle"]
    df["bb_upper"] = bb["bb_upper"]
    df["bb_lower"] = bb["bb_lower"]
    df["bb_width"] = bb["bb_width"]

    # ADX
    df["adx"] = adx(df["high"], df["low"], df["close"])
    df["adx_delta"] = df["adx"].diff()

    # 历史波动率
    df["vol_20"] = realized_vol(df["ret"], window=20)
    df["up_vol_20"], df["down_vol_20"] = upside_downside_vol(df["ret"], window=20)

    # GARCH
    df["garch_vol"] = compute_garch_vol(df["ret"])
    df["garch_vol_slope"] = df["garch_vol"].diff()

    # 用于趋势显著性和 Hurst
    df["log_close"] = np.log(df["close"])
    df["t_stat"] = rolling_linreg_tstat(df["log_close"], window=60)
    df["hurst"] = rolling_hurst(df["log_close"], window=128)

    return df


# 4. 趋势结构参数 & 分类
TREND_PARAMS_DEFAULT = {
    "adx_weak": 20.0,
    "adx_strong": 25.0,

    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "rsi_mid_low": 45.0,
    "rsi_mid_high": 55.0,

    "hurst_trend": 0.5,
    "hurst_random_low": 0.45,
    "hurst_random_high": 0.55,

    "tstat_sig": 2.0,
    "tstat_insig": 1.0,

    "bb_width_low": None,
    "bb_width_high": None,
}

# 给 classify_states 用（只会用到 adx_weak / adx_strong）
TREND_PARAMS = TREND_PARAMS_DEFAULT


def classify_trend_regime_row(row: pd.Series,
                              p: dict) -> str:
    """
    放宽版趋势结构分类：
    - 第一档：强趋势 T1/T3（条件较严格）
    - 第二档：中等趋势 T1/T3（条件放松）
    - T2：低波动震荡（弱 ADX + RSI 中性）
    - T4：高波动 / 反转过渡（中等 ADX + 高波动 + 突破）
    - 其它：UNK
    """
    adx_val = row["adx"]
    rsi_val = row["rsi"]
    t_stat = row["t_stat"]
    bb_width = row["bb_width"]
    bb_upper = row["bb_upper"]
    bb_lower = row["bb_lower"]
    price = row["close"]

    macd_val = row["macd_hist"]
    macd_cross_up = bool(row.get("macd_cross_up", False))
    macd_cross_down = bool(row.get("macd_cross_down", False))

    ma_short = row["ma_short"]
    ma_mid = row["ma_mid"]
    ma_long = row["ma_long"]

    # 多周期一致方向：5/20/60
    dir_up_multi = (ma_short > ma_mid) and (ma_mid > ma_long)
    dir_down_multi = (ma_short < ma_mid) and (ma_mid < ma_long)

    rsi_mid_band = (p["rsi_mid_low"] <= rsi_val <= p["rsi_mid_high"])

    #  第一档：强趋势（高质量 T1 / T3）
    if (
        dir_up_multi
        and adx_val > p["adx_strong"]
        and t_stat > 0
        and (macd_cross_up or macd_val > 0)
        and rsi_val >= p["rsi_mid_high"]   # ≥ 55
    ):
        return "T1"

    if (
        dir_down_multi
        and adx_val > p["adx_strong"]
        and t_stat < 0
        and (macd_cross_down or macd_val < 0)
        and rsi_val <= p["rsi_mid_low"]    # ≤ 45
    ):
        return "T3"

    #  第二档：中等趋势（放松版 T1 / T3）
    if (
        dir_up_multi
        and adx_val > p["adx_weak"]
        and t_stat > 0
        and (macd_val > 0)
        and rsi_val > 50
    ):
        return "T1"

    if (
        dir_down_multi
        and adx_val > p["adx_weak"]
        and t_stat < 0
        and (macd_val < 0)
        and rsi_val < 50
    ):
        return "T3"

    # T2：低波动震荡 
    if (
        adx_val < p["adx_weak"]
        and rsi_mid_band
    ):
        return "T2"

    #  T4：趋势反转 / 不稳定 
    if (
        (p["adx_weak"] <= adx_val <= p["adx_strong"])
        and rsi_mid_band
        and (
            (p["bb_width_high"] is not None and bb_width >= p["bb_width_high"])
            or (price >= bb_upper)
            or (price <= bb_lower)
        )
    ):
        return "T4"

    #  其它：保持 UNK，让 HMM/后处理去学 
    return "UNK"


def classify_trend_regimes(df: pd.DataFrame,
                           params: dict | None = None,
                           bb_q_low: float = 0.3,
                           bb_q_high: float = 0.7) -> pd.Series:
    """
    按框架生成趋势结构标签：{"T1","T2","T3","T4","UNK"}
    """
    if params is None:
        params = TREND_PARAMS_DEFAULT.copy()
    else:
        base = TREND_PARAMS_DEFAULT.copy()
        base.update(params)
        params = base

    bb_low, bb_high = calibrate_bb_width_thresholds(df["bb_width"],
                                                    q_low=bb_q_low,
                                                    q_high=bb_q_high)
    params["bb_width_low"] = bb_low
    params["bb_width_high"] = bb_high

    regimes = df.apply(classify_trend_regime_row, axis=1, p=params)
    return regimes


def debug_trend_conditions(df: pd.DataFrame,
                           params: dict | None = None,
                           bb_q_low: float = 0.3,
                           bb_q_high: float = 0.7) -> None:
    """
    用来查看各个条件（ADX / t值 / Hurst / RSI / 布林 / 多周期均线等）
    在样本中的命中率，以及 T1/T2/T3/T4 “理想型组合”各自有多少点。
    """
    if params is None:
        params = TREND_PARAMS_DEFAULT.copy()
    else:
        base = TREND_PARAMS_DEFAULT.copy()
        base.update(params)
        params = base

    df = df.copy()
    n = len(df)
    if n == 0:
        print("df 为空，无法调试")
        return

    # 重新算一下 bb_width 的分位数（和 classify_trend_regimes 一致的逻辑）
    bb_low, bb_high = calibrate_bb_width_thresholds(df["bb_width"],
                                                    q_low=bb_q_low,
                                                    q_high=bb_q_high)
    params["bb_width_low"] = bb_low
    params["bb_width_high"] = bb_high

    # 单个条件命中率 
    cond = {}

    cond["dir_up_multi"] = (df["ma_short"] > df["ma_mid"]) & (df["ma_mid"] > df["ma_long"])
    cond["dir_down_multi"] = (df["ma_short"] < df["ma_mid"]) & (df["ma_mid"] < df["ma_long"])

    cond["adx_strong"] = df["adx"] > params["adx_strong"]
    cond["adx_weak"] = df["adx"] < params["adx_weak"]
    cond["adx_mid"] = (df["adx"] >= params["adx_weak"]) & (df["adx"] <= params["adx_strong"])

    cond["t_up_sig"] = (df["t_stat"] > 0) & (df["t_stat"].abs() >= params["tstat_sig"])
    cond["t_down_sig"] = (df["t_stat"] < 0) & (df["t_stat"].abs() >= params["tstat_sig"])
    cond["t_insig"] = df["t_stat"].abs() <= params["tstat_insig"]

    cond["hurst_trendy"] = df["hurst"] > params["hurst_trend"]
    cond["hurst_random"] = (df["hurst"] >= params["hurst_random_low"]) & (df["hurst"] <= params["hurst_random_high"])

    cond["rsi_mid"] = (df["rsi"] >= params["rsi_mid_low"]) & (df["rsi"] <= params["rsi_mid_high"])
    cond["rsi_high"] = df["rsi"] > params["rsi_high"]
    cond["rsi_low"] = df["rsi"] < params["rsi_low"]

    cond["bb_narrow"] = df["bb_width"] < params["bb_width_low"]
    cond["bb_wide"] = df["bb_width"] >= params["bb_width_high"]

    cond["price_break_upper"] = df["close"] >= df["bb_upper"]
    cond["price_break_lower"] = df["close"] <= df["bb_lower"]

    cond["macd_pos_or_cross_up"] = (df["macd_hist"] > 0) | (df["macd_cross_up"])
    cond["macd_neg_or_cross_down"] = (df["macd_hist"] < 0) | (df["macd_cross_down"])

    print("==== 单个条件命中率（占全部样本的百分比）====")
    for name, m in cond.items():
        ratio = float(m.mean()) if len(m) > 0 else 0.0
        print(f"{name:25s}: {ratio:6.2%}")

    # 组合条件：理想 T1/T3/T2/T4 覆盖率 
    # T1 理想型
    mask_T1 = (
        cond["dir_up_multi"]
        & cond["adx_strong"]
        & cond["t_up_sig"]
        & cond["hurst_trendy"]
        & cond["macd_pos_or_cross_up"]
        & cond["rsi_high"]
        & cond["price_break_upper"]
    )

    # T3 理想型
    mask_T3 = (
        cond["dir_down_multi"]
        & cond["adx_strong"]
        & cond["t_down_sig"]
        & cond["hurst_trendy"]
        & cond["macd_neg_or_cross_down"]
        & cond["rsi_low"]
        & cond["price_break_lower"]
    )

    # T2 理想型
    mask_T2 = (
        cond["adx_weak"]
        & cond["t_insig"]
        & cond["hurst_random"]
        & cond["bb_narrow"]
        & cond["rsi_mid"]
    )

    # T4 理想型
    mask_T4 = (
        cond["adx_mid"]
        & cond["t_insig"]
        & cond["hurst_random"]
        & cond["bb_wide"]
        & cond["rsi_mid"]
        & (cond["price_break_upper"] | cond["price_break_lower"])
    )

    print("\n==== 理想型组合命中率（单独统计，不看最终 trend_regime）====")
    print(f"T1 ideal mask: {mask_T1.mean():6.2%}")
    print(f"T3 ideal mask: {mask_T3.mean():6.2%}")
    print(f"T2 ideal mask: {mask_T2.mean():6.2%}")
    print(f"T4 ideal mask: {mask_T4.mean():6.2%}")

    #  最终标签的分布
    if "trend_regime" in df.columns:
        print("\n==== 最终 trend_regime 标签分布 ==== ")
        print(df["trend_regime"].value_counts(dropna=False, normalize=True).map(lambda x: f"{x:6.2%}"))
    else:
        print("\n提示：df 里还没有 trend_regime 列，可在 classify_trend_regimes 之后再调试。")


# 5. 波动率结构 V1~V4
VOL_PARAMS = {
    "dg_small": 0.0005,      # GARCH 斜率“很小”的阈值

    "rsi_mid_low": 45.0,
    "rsi_mid_high": 55.0,
    "rsi_extreme_low": 30.0,
    "rsi_extreme_high": 70.0,

    "low_q": 0.2,
    "high_q": 0.8,
}


def make_vol_classifier(df: pd.DataFrame,
                        params: dict = VOL_PARAMS):
    """
    构造 V1~V4 分类器函数（基于整段数据的分布）
    """

    df = df.copy()

    g = df["garch_vol"].dropna()
    if len(g) == 0:
        raise ValueError("garch_vol is empty, check compute_garch_vol / data length")

    low_q = params.get("low_q", 0.2)
    high_q = params.get("high_q", 0.8)

    g_low = g.quantile(low_q)
    g_high = g.quantile(high_q)

    rv = df["vol_20"].dropna()
    bw = df["bb_width"].dropna()

    if len(rv) > 0:
        rv_low = rv.quantile(0.3)
        rv_high = rv.quantile(0.7)
    else:
        rv_low = rv_high = np.nan

    if len(bw) > 0:
        bw_low = bw.quantile(0.3)
        bw_high = bw.quantile(0.7)
    else:
        bw_low = bw_high = np.nan

    rsi_mid_low = params["rsi_mid_low"]
    rsi_mid_high = params["rsi_mid_high"]
    rsi_extreme_low = params["rsi_extreme_low"]
    rsi_extreme_high = params["rsi_extreme_high"]
    dg_small = params["dg_small"]

    def classify_vol_level(row: pd.Series) -> str:
        g_val = row["garch_vol"]
        dg = row["garch_vol_slope"]
        rv_val = row["vol_20"]
        bw_val = row["bb_width"]
        rsi_val = row["rsi"]
        price = row["close"]
        bb_upper = row["bb_upper"]
        bb_lower = row["bb_lower"]

        # 缺 garch 信息的回退逻辑
        if pd.isna(g_val) or pd.isna(dg):
            if (
                (not pd.isna(rv_val) and rv_val <= rv_low)
                and (not pd.isna(bw_val) and bw_val <= bw_low)
                and (rsi_mid_low <= rsi_val <= rsi_mid_high)
            ):
                return "V1"

            if (
                (not pd.isna(rv_val) and rv_val >= rv_high)
                and (not pd.isna(bw_val) and bw_val >= bw_high)
                and (
                    rsi_val <= rsi_extreme_low
                    or rsi_val >= rsi_extreme_high
                    or price >= bb_upper
                    or price <= bb_lower
                )
            ):
                return "V3"

            return "V2"

        rsi_mid = (rsi_mid_low <= rsi_val <= rsi_mid_high)
        rsi_extreme = (rsi_val <= rsi_extreme_low) or (rsi_val >= rsi_extreme_high)

        # V1: 低波动收敛
        if (
            (g_val <= g_low)
            and (dg <= 0)
            and (not pd.isna(rv_val) and rv_val <= rv_low)
            and (not pd.isna(bw_val) and bw_val <= bw_low)
            and rsi_mid
        ):
            return "V1"

        # 高波动区：V3 / V4
        if g_val >= g_high:
            # V3: 高波动发散
            if (
                dg > dg_small
                and (not pd.isna(bw_val) and bw_val >= bw_high)
                and (
                    rsi_extreme
                    or price >= bb_upper
                    or price <= bb_lower
                )
            ):
                return "V3"

            # V4: 高波动衰减
            if (
                dg < -dg_small
                and (pd.isna(rv_high) or rv_val <= rv_high)
                and rsi_mid
            ):
                return "V4"

            if abs(dg) <= dg_small:
                return "V3"

        # V2: 中波动稳定
        if (
            (g_val > g_low) and (g_val < g_high)
            and abs(dg) <= dg_small
            and (pd.isna(bw_low) or pd.isna(bw_high) or (bw_low <= bw_val <= bw_high))
            and rsi_mid
        ):
            return "V2"

        # 兜底
        if dg > 0:
            return "V3"
        else:
            if g_val >= g_low:
                return "V4"
            else:
                return "V1"

    return classify_vol_level


# 6. 路径依赖结构 S1~S5
def classify_states(df: pd.DataFrame) -> pd.Series:
    """
    基于 trend_regime, vol_regime, adx, adx_delta 给出 S1~S5：
    S1: 趋势启动期
    S2: 趋势延续期
    S3: 趋势衰竭期
    S4: 高波动震荡
    S5: 恢复/修复期
    """
    states: list[str] = []
    prev_state: str | None = None

    for _, row in df.iterrows():
        T = row["trend_regime"]
        V = row["vol_regime"]
        adx_val = row["adx"]
        adx_delta = row["adx_delta"]

        state = "S5"  # 默认：恢复/修复

        if T in ("T1", "T3"):
            # 启动期：从 S5/S4 刚切到强趋势，ADX 从低位上升
            if (
                prev_state in (None, "S5", "S4")
                and adx_val > TREND_PARAMS["adx_weak"]
                and adx_delta > 0
            ):
                state = "S1"
            # 延续期：ADX 较高且变化不大
            elif adx_val > TREND_PARAMS["adx_strong"] and abs(adx_delta) < 1:
                state = "S2"
            # 衰竭期：ADX 回落 + 高波动
            elif adx_delta < 0 and V in ("V3", "V4"):
                state = "S3"
            else:
                state = "S2"

        elif T in ("T2", "T4") and V in ("V3", "V4"):
            # 高波动震荡
            state = "S4"

        else:
            # 震荡 + 波动回落 → 修复
            state = "S5"

        states.append(state)
        prev_state = state

    return pd.Series(states, index=df.index)

def compute_states_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入:
        df: 至少包含 ['date','open','high','low','close'] 列（行情数据）
    输出:
        一个新的 df_feat，在原 df 基础上加上:
            - trend_regime  (T1~T4/UNK)
            - vol_regime    (V1~V4)
            - state         (S1~S5)
            - hmm_state, hmm_state_label（如果 HMM 拟合成功）
    """
    df = df.copy()

    # 确保日期是 datetime，按日期排序
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # 4. 计算特征
    df_feat = compute_features(df)

    # 5. 趋势结构
    df_feat["trend_regime"] = classify_trend_regimes(df_feat)

    # 6. 波动结构
    vol_classifier = make_vol_classifier(df_feat)
    df_feat["vol_regime"] = df_feat.apply(vol_classifier, axis=1)

    # 7. 路径依赖状态 S1~S5
    df_feat["state"] = classify_states(df_feat)

    # 7.2 HMM（用 hmmlearn 自动学）
    feature_cols = ["ret", "adx", "bb_width", "vol_20", "rsi", "macd_hist"]
    df_hmm = df_feat.dropna(subset=feature_cols).copy()
    X = df_hmm[feature_cols].values

    try:
        if len(df_hmm) < 10:
            raise ValueError("用于 HMM 的样本太少（<10），请检查数据。")

        n_states = 5
        model_hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )

        model_hmm.fit(X)
        hmm_states = model_hmm.predict(X)

        df_feat.loc[df_hmm.index, "hmm_state"] = hmm_states

        # 映射到规则状态标签
        mapping = map_hmm_state_to_rule_state(df_feat, hmm_col="hmm_state", rule_col="state")
        df_feat["hmm_state_label"] = df_feat["hmm_state"].map(mapping)

    except Exception as e:
        print("[WARN] HMM 拟合或解码失败：", e)

    return df_feat


def estimate_transition_matrix(states: pd.Series,
                               state_order: list[str] | None = None) -> pd.DataFrame:
    """
    根据状态序列（如 S1~S5）估计一阶马尔可夫转移矩阵 P(S_t | S_{t-1})。

    参数：
    - states: 一维序列，如 df["state"] 或 df["trend_regime"]
    - state_order: 想固定的状态顺序，例如 ["S1","S2","S3","S4","S5"]

    返回：
    - DataFrame，行是 from_state，列是 to_state，每行按行归一化为 1
    """
    s = states.astype(str)

    from_state = s.shift(1)   # S_{t-1}
    to_state = s              # S_t

    # 交叉表 + 按行归一化 -> P(S_t | S_{t-1})
    trans = pd.crosstab(from_state, to_state, normalize="index")

    # 如果指定了状态顺序，补齐缺失状态，用 0 填
    if state_order is not None:
        trans = trans.reindex(index=state_order, columns=state_order, fill_value=0.0)

    return trans


def map_hmm_state_to_rule_state(df: pd.DataFrame,
                                hmm_col: str = "hmm_state",
                                rule_col: str = "state") -> dict[int, str]:
    """
    对每个 HMM 隐状态 k，统计它对应的规则状态出现频率最高的是谁，
    返回映射：k -> "S1"/"S2"/...
    """
    mapping: dict[int, str] = {}
    sub = df.dropna(subset=[hmm_col, rule_col]).copy()
    if sub.empty:
        return mapping

    sub[hmm_col] = sub[hmm_col].astype(int)
    sub[rule_col] = sub[rule_col].astype(str)

    for k in sorted(sub[hmm_col].unique()):
        counts = sub.loc[sub[hmm_col] == k, rule_col].value_counts()
        if len(counts) == 0:
            continue
        mapping[k] = counts.idxmax()
    return mapping


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    """
    兼容 utf-8 / gbk 编码的中文 CSV
    """
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def main() -> None:
    # 1. 读入原始 CSV
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")

    df_raw = load_csv_with_fallback(INPUT_CSV)

    # 2. 重命名列
    df = df_raw.rename(columns=COLUMN_MAP).copy()

    # 3. 日期处理 & 排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 4. 计算特征
    df_feat = compute_features(df)

    # 5. 趋势结构
    df_feat["trend_regime"] = classify_trend_regimes(df_feat)

    print("\n========== 趋势条件调试输出 ==========")
    debug_trend_conditions(df_feat)

    # 6. 波动结构
    vol_classifier = make_vol_classifier(df_feat)
    df_feat["vol_regime"] = df_feat.apply(vol_classifier, axis=1)

    # 7. 路径依赖状态 S1~S5
    df_feat["state"] = classify_states(df_feat)

    # 7.1 估计 S1~S5 的状态转移概率矩阵
    P_state = estimate_transition_matrix(
        df_feat["state"],
        state_order=["S1", "S2", "S3", "S4", "S5"]
    )

    print("\n===== 状态转移矩阵 P(S_t | S_{t-1}) =====")
    print(P_state.round(3))   # 保留三位小数方便看

    # 趋势 T1~T4 的转移矩阵
    P_trend = estimate_transition_matrix(
        df_feat["trend_regime"],
        state_order=["T1", "T2", "T3", "T4", "UNK"]
    )
    print("\n===== 趋势结构转移矩阵 P(T_t | T_{t-1}) =====")
    print(P_trend.round(3))

    # 7.2 使用 hmmlearn 拟合 GaussianHMM（不手动初始化）
    feature_cols = ["ret", "adx", "bb_width", "vol_20", "rsi", "macd_hist"]

    df_hmm = df_feat.dropna(subset=feature_cols).copy()
    X = df_hmm[feature_cols].values

    try:
        if len(df_hmm) < 10:
            raise ValueError("用于 HMM 的样本太少（<10），请检查数据长度或特征缺失情况。")

        n_states = 5  # 隐状态个数，和 S1~S5 数量对应
        model_hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )

        # EM 训练 + 预测隐状态
        model_hmm.fit(X)
        hmm_states = model_hmm.predict(X)

        # 写回 df_feat
        df_feat.loc[df_hmm.index, "hmm_state"] = hmm_states

        print("\n===== HMM 隐状态分布（0 ~ n_states-1） =====")
        print(df_feat["hmm_state"].value_counts(normalize=True))

        print("\n===== HMM 转移矩阵（model_hmm.transmat_） =====")
        print(pd.DataFrame(
            model_hmm.transmat_,
            index=[f"Z{i}" for i in range(n_states)],
            columns=[f"Z{i}" for i in range(n_states)]
        ).round(3))

        # HMM 状态 -> 规则状态 的解释映射（可选）
        mapping = map_hmm_state_to_rule_state(df_feat, hmm_col="hmm_state", rule_col="state")
        print("\n===== HMM 状态 -> 规则状态 映射 =====")
        print(mapping)

        # 把映射后的标签写回
        df_feat["hmm_state_label"] = df_feat["hmm_state"].map(mapping)

    except Exception as e:
        print("\n[WARN] HMM 拟合或解码失败：", e)

    # 8. 导出结果
    cols_out = [
        "date", "open", "high", "low", "close",
        "trend_regime", "vol_regime", "state"
    ]
    # 如果 HMM 跑成功，会多两列
    if "hmm_state" in df_feat.columns:
        cols_out += ["hmm_state", "hmm_state_label"]

    df_out = df_feat[cols_out].copy()

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved classified data to:", OUTPUT_CSV)
    print(df_out.tail())


if __name__ == "__main__":
    main()
