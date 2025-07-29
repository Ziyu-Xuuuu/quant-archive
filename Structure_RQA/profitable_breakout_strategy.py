# -*- coding: utf-8 -*-
"""
Profitable Breakout Strategy 2025 v1.0
=======================================
高频突破均值回归策略，专注于捕捉短期价格失衡机会

核心逻辑：
1. 多时间框架分析 - 1日、3日、5日时间框架结合
2. 波动率突破 - 识别价格异常波动后的均值回归机会
3. 量价配合 - 成交量确认突破有效性
4. 快速止盈止损 - 短期持有，降低市场风险
5. 智能仓位管理 - 根据市场状态动态调整仓位
"""

from rqalpha.api import (
    all_instruments,
    history_bars,
    order_target_percent,
    get_trading_dates,
)
import numpy as np
import pandas as pd

__config__ = {
    "base": {
        "start_date": "2025-01-01",
        "end_date": "2025-07-01", 
        "frequency": "1d",
        "accounts": {"stock": 1_000_000},
        "benchmark": "000300.XSHG",
    },
    "extra": {"log_level": "info"},
}

def init(context):
    """策略初始化"""
    # 核心参数
    context.max_positions = 8          # 最大持仓8只
    context.base_position = 0.90       # 基础仓位90%
    context.single_position_limit = 0.15  # 单股最大15%
    
    # 技术指标参数
    context.volatility_window = 20     # 波动率计算窗口
    context.volatility_threshold = 2.0 # 波动率突破阈值
    context.rsi_period = 14           # RSI周期
    context.rsi_oversold = 25         # RSI超卖阈值
    context.rsi_overbought = 75       # RSI超买阈值
    
    # 突破参数
    context.breakout_period = 10      # 突破周期
    context.volume_multiplier = 1.8   # 成交量放大倍数
    context.price_change_threshold = 0.03  # 价格变化阈值
    
    # 风控参数
    context.stop_loss = 0.06          # 6%止损
    context.take_profit_fast = 0.08   # 快速8%止盈
    context.take_profit_slow = 0.15   # 慢速15%止盈
    context.max_hold_days = 8         # 最大持有8天
    context.daily_loss_limit = 0.02   # 单日最大损失2%
    
    # 筛选条件
    context.min_price = 8.0
    context.max_price = 100.0
    context.min_market_cap = 50e8     # 最小市值50亿
    context.min_volume_value = 80e6   # 日成交额8000万+
    
    # 市场状态参数
    context.market_volatility_period = 15
    context.bull_market_threshold = 0.02
    context.bear_market_threshold = -0.02
    
    # 状态记录
    context.positions = {}
    context.stock_universe = set()
    context.market_state = 'neutral'
    context.daily_returns = []
    context.performance_tracker = []
    context.scan_frequency = 1        # 每天扫描
    context.last_scan = None
    context.debug = True
    
    # 初始化股票池
    initialize_stock_universe(context)
    print(f"Profitable Breakout Strategy 初始化完成 - 股票池: {len(context.stock_universe)}只")

def initialize_stock_universe(context):
    """初始化股票池"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 排除特殊股票
        if any(x in symbol for x in ['ST', '*ST', '退', 'N']):
            continue
            
        # 排除科创板和创业板的高风险股票（可选）
        if stock.startswith('688') or stock.startswith('300'):
            continue
            
        universe.add(stock)
    
    context.stock_universe = universe

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 更新市场状态
    update_market_state(context)
    
    # 记录表现
    nav = context.portfolio.total_value
    daily_return = 0
    if context.performance_tracker:
        prev_nav = context.performance_tracker[-1]['nav']
        daily_return = (nav / prev_nav - 1)
    
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'daily_return': daily_return,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0]),
        'market_state': context.market_state
    })
    
    # 风控检查 - 如果当日损失过大，暂停交易
    if daily_return < -context.daily_loss_limit:
        if context.debug:
            print(f"{current_date} 触发日损失限制 {daily_return:.2%}，暂停新开仓")
        manage_existing_positions(context, bar_dict, current_date)
        return
    
    # 管理现有持仓
    manage_existing_positions(context, bar_dict, current_date)
    
    # 扫描新机会
    if should_scan_opportunities(context, current_date):
        scan_breakout_opportunities(context, bar_dict, current_date)

def update_market_state(context):
    """更新市场状态"""
    try:
        # 获取基准指数数据
        benchmark_data = history_bars('000300.XSHG', context.market_volatility_period + 5, '1d', 'close', skip_suspended=True, include_now=True)
        
        if benchmark_data is None or len(benchmark_data) < context.market_volatility_period:
            context.market_state = 'neutral'
            return
        
        # 计算市场趋势
        recent_return = (benchmark_data[-1] / benchmark_data[-context.market_volatility_period] - 1)
        
        if recent_return > context.bull_market_threshold:
            context.market_state = 'bull'
        elif recent_return < context.bear_market_threshold:
            context.market_state = 'bear'
        else:
            context.market_state = 'neutral'
            
    except:
        context.market_state = 'neutral'

def manage_existing_positions(context, bar_dict, current_date):
    """管理现有持仓"""
    to_close = []
    
    for stock in list(context.positions.keys()):
        position = context.portfolio.positions.get(stock)
        if not position or position.quantity <= 0:
            context.positions.pop(stock, None)
            continue
        
        pos_info = context.positions[stock]
        current_price = bar_dict[stock].close
        entry_price = pos_info['entry_price']
        entry_date = pos_info['entry_date']
        strategy_type = pos_info.get('strategy_type', 'breakout')
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        # 退出条件判断
        should_close = False
        reason = ""
        
        # 止盈止损
        if return_pct >= context.take_profit_slow:
            should_close, reason = True, f"慢速止盈，收益: {return_pct:.2%}"
        elif return_pct >= context.take_profit_fast and strategy_type == 'mean_reversion':
            should_close, reason = True, f"快速止盈，收益: {return_pct:.2%}"
        elif return_pct <= -context.stop_loss:
            should_close, reason = True, f"止损，损失: {return_pct:.2%}"
        elif hold_days >= context.max_hold_days:
            should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
        elif check_exit_signal(stock, context, strategy_type):
            should_close, reason = True, f"技术信号退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason))
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_exit_signal(stock, context, strategy_type):
    """检查技术性退出信号"""
    try:
        closes = history_bars(stock, 15, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 10, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 10:
            return False
        
        if strategy_type == 'breakout':
            # 突破策略退出：连续下跌或成交量萎缩
            recent_changes = np.diff(closes[-4:])
            if np.sum(recent_changes < 0) >= 3:
                return True
            
            # 成交量萎缩超过60%
            vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-7:])
            if vol_ratio < 0.4:
                return True
                
        elif strategy_type == 'mean_reversion':
            # 均值回归策略退出：RSI过度超买
            rsi = calculate_rsi(closes, context.rsi_period)
            if rsi > context.rsi_overbought:
                return True
        
        return False
    except:
        return False

def should_scan_opportunities(context, current_date):
    """判断是否需要扫描机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    if current_positions < context.max_positions:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= 1  # 未满仓每天扫描
    else:
        return False  # 满仓不扫描

def scan_breakout_opportunities(context, bar_dict, current_date):
    """扫描突破机会"""
    if context.debug:
        print(f"{current_date} 扫描突破机会 - 市场状态: {context.market_state}")
    
    breakout_candidates = []
    mean_reversion_candidates = []
    
    # 限制扫描数量以提高效率
    scan_stocks = list(context.stock_universe)[:300]
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            # 基础筛选
            if not basic_stock_filter(stock, bar_dict, context):
                continue
            
            # 突破信号检测
            breakout_score = detect_breakout_signal(stock, context)
            if breakout_score > 0.7:
                breakout_candidates.append((stock, breakout_score, 'breakout'))
            
            # 均值回归信号检测
            reversion_score = detect_mean_reversion_signal(stock, context)
            if reversion_score > 0.7:
                mean_reversion_candidates.append((stock, reversion_score, 'mean_reversion'))
        
        except:
            continue
    
    # 合并候选股票
    all_candidates = breakout_candidates + mean_reversion_candidates
    
    if not all_candidates:
        if context.debug:
            print(f"{current_date} 未发现交易机会")
        context.last_scan = current_date
        return
    
    # 按评分排序
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 根据市场状态调整选择偏好
    if context.market_state == 'bull':
        # 牛市偏好突破策略
        selected_candidates = [c for c in all_candidates if c[2] == 'breakout'][:4] + \
                            [c for c in all_candidates if c[2] == 'mean_reversion'][:2]
    elif context.market_state == 'bear':
        # 熊市偏好均值回归
        selected_candidates = [c for c in all_candidates if c[2] == 'mean_reversion'][:4] + \
                            [c for c in all_candidates if c[2] == 'breakout'][:2]
    else:
        # 震荡市均衡配置
        selected_candidates = all_candidates[:6]
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    for i, (stock, score, strategy_type) in enumerate(selected_candidates[:available_slots]):
        position_size = calculate_position_size(context, score, strategy_type, i)
        
        if position_size > 0.03:  # 最小3%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'strategy_type': strategy_type,
                'score': score
            }
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 策略: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.2f}")
    
    context.last_scan = current_date

def basic_stock_filter(stock, bar_dict, context):
    """基础股票筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取历史数据
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-10:] * volumes[-10:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 避免连续大跌股票
        recent_decline = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        if recent_decline < -0.15:
            return False
        
        # 避免涨停板
        daily_return = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        if daily_return > 0.095:  # 接近涨停
            return False
        
        return True
    
    except:
        return False

def detect_breakout_signal(stock, context):
    """检测突破信号"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        highs = history_bars(stock, 15, '1d', 'high', skip_suspended=True, include_now=True)
        lows = history_bars(stock, 15, '1d', 'low', skip_suspended=True, include_now=True)
        
        if any(x is None for x in [closes, volumes, highs, lows]) or len(closes) < 20:
            return 0
        
        current_price = closes[-1]
        score = 0
        
        # 1. 价格突破信号 (40%)
        breakout_high = np.max(highs[-context.breakout_period:-1])  # 排除当天
        if current_price > breakout_high * 1.02:  # 突破前期高点2%以上
            score += 0.4
        elif current_price > breakout_high:
            score += 0.25
        
        # 2. 成交量确认 (30%)
        volume_avg = np.mean(volumes[-10:-1])
        current_volume = volumes[-1]
        if current_volume > volume_avg * context.volume_multiplier:
            score += 0.3
        elif current_volume > volume_avg * 1.3:
            score += 0.2
        
        # 3. 波动率突破 (20%)
        volatility = calculate_volatility(closes, context.volatility_window)
        avg_volatility = np.mean([calculate_volatility(closes[i:i+context.volatility_window], context.volatility_window) 
                                for i in range(5) if len(closes[i:i+context.volatility_window]) == context.volatility_window])
        
        if volatility > avg_volatility * context.volatility_threshold:
            score += 0.2
        elif volatility > avg_volatility * 1.5:
            score += 0.1
        
        # 4. 技术形态 (10%)
        if len(closes) >= 15:
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            if current_price > ma5 > ma10:
                score += 0.1
            elif current_price > ma5:
                score += 0.05
        
        return min(score, 1.0)
    
    except:
        return 0

def detect_mean_reversion_signal(stock, context):
    """检测均值回归信号"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return 0
        
        current_price = closes[-1]
        score = 0
        
        # 1. RSI超卖信号 (40%)
        rsi = calculate_rsi(closes, context.rsi_period)
        if rsi < context.rsi_oversold:
            oversold_intensity = (context.rsi_oversold - rsi) / context.rsi_oversold
            score += 0.4 * min(oversold_intensity * 2, 1.0)
        elif rsi < 35:
            score += 0.2
        
        # 2. 价格偏离均线 (30%)
        ma20 = np.mean(closes[-20:])
        price_deviation = (current_price / ma20 - 1)
        if price_deviation < -0.08:  # 价格低于20日均线8%以上
            score += 0.3
        elif price_deviation < -0.05:
            score += 0.2
        elif price_deviation < -0.03:
            score += 0.1
        
        # 3. 反弹确认信号 (20%)
        recent_low = np.min(closes[-5:])
        if current_price > recent_low * 1.02:  # 开始反弹
            score += 0.2
        elif current_price > recent_low * 1.01:
            score += 0.1
        
        # 4. 成交量配合 (10%)
        volume_avg = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        if current_volume > volume_avg * 1.2:
            score += 0.1
        elif current_volume > volume_avg:
            score += 0.05
        
        return min(score, 1.0)
    
    except:
        return 0

def calculate_volatility(prices, window):
    """计算价格波动率"""
    if len(prices) < window:
        return 0
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns[-window:])

def calculate_rsi(prices, period):
    """计算RSI指标"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_position_size(context, score, strategy_type, position_index):
    """计算仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 根据评分调整
    score_multiplier = 0.6 + 0.6 * score
    
    # 根据策略类型调整
    if context.market_state == 'bull' and strategy_type == 'breakout':
        strategy_multiplier = 1.2
    elif context.market_state == 'bear' and strategy_type == 'mean_reversion':
        strategy_multiplier = 1.2
    else:
        strategy_multiplier = 1.0
    
    # 首选股票获得更大仓位
    priority_multiplier = 1.3 if position_index == 0 else (1.1 if position_index == 1 else 1.0)
    
    # 计算最终仓位
    position_size = base_size * score_multiplier * strategy_multiplier * priority_multiplier
    
    # 限制单股最大仓位
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0)