# -*- coding: utf-8 -*-
"""
Final Profitable Strategy 2025 v3.0
====================================
基于前两次回测优化的最终高收益策略

关键优化：
1. 专注最成功的交易模式
2. 更严格的入场条件
3. 更精确的止盈止损机制
4. 更好的资金管理
5. 减少交易频率，提高胜率
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
    # 核心参数 - 精英化配置
    context.max_positions = 4          # 集中持仓4只精品
    context.base_position = 0.96       # 几乎满仓操作
    context.single_position_limit = 0.30  # 允许更大单股仓位
    
    # 技术指标参数 - 精确调优
    context.volatility_window = 12
    context.volatility_threshold = 2.8
    context.rsi_period = 8
    context.rsi_oversold = 15
    context.rsi_overbought = 85
    
    # 突破参数 - 高质量信号
    context.breakout_period = 6
    context.volume_multiplier = 2.5
    context.momentum_threshold = 0.06
    context.price_acceleration_min = 0.03
    
    # 风控参数 - 精细化控制
    context.stop_loss = 0.04          # 严格4%止损
    context.take_profit_fast = 0.12   # 快速12%止盈
    context.take_profit_medium = 0.20 # 中等20%止盈
    context.take_profit_high = 0.35   # 高收益35%止盈
    context.max_hold_days = 5         # 极短持有期
    context.daily_loss_limit = 0.012  # 严格日损失控制
    
    # 筛选条件 - 最严格标准
    context.min_price = 12.0
    context.max_price = 70.0
    context.min_market_cap = 100e8    # 100亿市值以上
    context.min_volume_value = 150e6  # 1.5亿成交额
    context.min_days_listed = 252     # 至少上市1年
    
    # 质量评分阈值 - 只选最优
    context.min_total_score = 0.92
    context.min_quality_score = 0.85
    context.min_momentum_score = 0.80
    context.min_volume_score = 0.75
    
    # 市场状态参数
    context.bull_threshold = 0.01
    context.bear_threshold = -0.01
    
    # 状态记录
    context.positions = {}
    context.elite_stock_pool = set()
    context.market_state = 'neutral'
    context.performance_tracker = []
    context.scan_frequency = 2        # 每2天扫描一次，提高质量
    context.last_scan = None
    context.debug = True
    context.total_trades = 0
    context.winning_trades = 0
    
    # 初始化精英股票池
    initialize_elite_universe(context)
    print(f"Final Profitable Strategy v3.0 初始化完成 - 精英股票池: {len(context.elite_stock_pool)}只")

def initialize_elite_universe(context):
    """初始化精英股票池"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 只选择最优质股票
        if any(x in symbol for x in ['ST', '*ST', '退', 'N', 'C']):
            continue
            
        # 只选择主板股票
        if stock.startswith('688') or stock.startswith('300') or stock.startswith('301'):
            continue
            
        # 主要选择上证和深圳主板
        if not (stock.endswith('.XSHG') or (stock.endswith('.XSHE') and stock.startswith('00'))):
            continue
            
        universe.add(stock)
    
    context.elite_stock_pool = universe

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
        'market_state': context.market_state,
        'total_trades': context.total_trades,
        'win_rate': context.winning_trades / max(context.total_trades, 1) * 100
    })
    
    # 风控检查
    if daily_return < -context.daily_loss_limit:
        if context.debug:
            print(f"{current_date} 触发风控，暂停交易")
        manage_elite_positions(context, bar_dict, current_date, risk_mode=True)
        return
    
    # 管理现有持仓
    manage_elite_positions(context, bar_dict, current_date)
    
    # 扫描新机会 - 降低频率，提高质量
    if should_scan_elite_opportunities(context, current_date):
        scan_elite_opportunities(context, bar_dict, current_date)

def update_market_state(context):
    """更新市场状态"""
    try:
        benchmark_data = history_bars('000300.XSHG', 15, '1d', 'close', skip_suspended=True, include_now=True)
        
        if benchmark_data is None or len(benchmark_data) < 10:
            context.market_state = 'neutral'
            return
        
        # 短中期综合趋势
        short_trend = (benchmark_data[-1] / benchmark_data[-3] - 1)
        medium_trend = (benchmark_data[-1] / benchmark_data[-8] - 1)
        combined_trend = short_trend * 0.7 + medium_trend * 0.3
        
        if combined_trend > context.bull_threshold and short_trend > 0:
            context.market_state = 'bull'
        elif combined_trend < context.bear_threshold and short_trend < 0:
            context.market_state = 'bear'  
        else:
            context.market_state = 'neutral'
            
    except:
        context.market_state = 'neutral'

def manage_elite_positions(context, bar_dict, current_date, risk_mode=False):
    """管理精英持仓"""
    to_close = []
    to_reduce = []
    
    for stock in list(context.positions.keys()):
        position = context.portfolio.positions.get(stock)
        if not position or position.quantity <= 0:
            context.positions.pop(stock, None)
            continue
        
        pos_info = context.positions[stock]
        current_price = bar_dict[stock].close
        entry_price = pos_info['entry_price']
        entry_date = pos_info['entry_date']
        profit_level = pos_info.get('profit_level', 0)
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        # 分级止盈止损策略
        should_close = False
        should_reduce = False
        reason = ""
        
        if risk_mode:
            if return_pct <= -context.stop_loss * 0.7:
                should_close, reason = True, f"风控止损，损失: {return_pct:.2%}"
        else:
            # 分级止盈
            if profit_level == 0 and return_pct >= context.take_profit_fast:
                should_reduce, reason = True, f"一层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 1
            elif profit_level == 1 and return_pct >= context.take_profit_medium:
                should_reduce, reason = True, f"二层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 2
            elif return_pct >= context.take_profit_high:
                should_close, reason = True, f"高收益止盈，收益: {return_pct:.2%}"
            elif return_pct <= -context.stop_loss:
                should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            elif hold_days >= context.max_hold_days:
                should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            elif check_elite_exit_signal(stock, context):
                should_close, reason = True, f"技术退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason, return_pct))
        elif should_reduce:
            to_reduce.append((stock, reason))
    
    # 执行减仓
    for stock, reason in to_reduce:
        position = context.portfolio.positions.get(stock)
        current_weight = position.market_value / context.portfolio.total_value
        new_weight = current_weight * 0.6  # 减仓至60%
        order_target_percent(stock, new_weight)
        if context.debug:
            print(f"{current_date} 减仓 {stock}: {reason}")
    
    # 执行平仓并统计
    for stock, reason, return_pct in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        context.total_trades += 1
        if return_pct > 0:
            context.winning_trades += 1
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_elite_exit_signal(stock, context):
    """检查精英退出信号"""
    try:
        closes = history_bars(stock, 10, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 8, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 8:
            return False
        
        # 严格的退出条件
        # 连续2天下跌超过3%
        if len(closes) >= 3:
            decline_2d = (closes[-1] / closes[-3] - 1)
            if decline_2d < -0.03:
                return True
        
        # 成交量极度萎缩
        if len(volumes) >= 6:
            vol_ratio = np.mean(volumes[-2:]) / np.mean(volumes[-6:])
            if vol_ratio < 0.25:
                return True
        
        return False
    except:
        return False

def should_scan_elite_opportunities(context, current_date):
    """判断是否扫描精英机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    if current_positions >= context.max_positions:
        return False
    
    if context.last_scan is None:
        return True
        
    days_since_scan = (current_date - context.last_scan).days
    return days_since_scan >= context.scan_frequency

def scan_elite_opportunities(context, bar_dict, current_date):
    """扫描精英机会"""
    if context.debug:
        print(f"{current_date} 扫描精英机会 - 市场: {context.market_state}")
    
    candidates = []
    scan_stocks = list(context.elite_stock_pool)[:150]  # 限制扫描范围
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            if not elite_stock_filter(stock, bar_dict, context):
                continue
            
            # 三维评分系统
            quality_score = calculate_elite_quality_score(stock, context)
            momentum_score = calculate_elite_momentum_score(stock, context)
            volume_score = calculate_elite_volume_score(stock, context)
            
            # 必须所有维度都达标
            if (quality_score >= context.min_quality_score and 
                momentum_score >= context.min_momentum_score and 
                volume_score >= context.min_volume_score):
                
                # 综合评分
                total_score = (quality_score * 0.4 + momentum_score * 0.4 + volume_score * 0.2)
                
                if total_score >= context.min_total_score:
                    strategy_type = 'breakout' if momentum_score > quality_score else 'quality'
                    candidates.append((stock, total_score, strategy_type))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现精英级机会")
        context.last_scan = current_date
        return
    
    # 按评分排序，只选最优
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓 - 高度集中
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    for i, (stock, score, strategy_type) in enumerate(candidates[:available_slots]):
        position_size = calculate_elite_position_size(context, score, i)
        
        if position_size > 0.08:  # 最小8%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'strategy_type': strategy_type,
                'score': score,
                'profit_level': 0
            }
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 类型: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.3f}")
    
    context.last_scan = current_date

def elite_stock_filter(stock, bar_dict, context):
    """精英股票筛选"""
    try:
        price = bar_dict[stock].close
        
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取更长期数据
        closes = history_bars(stock, 60, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 40, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 30:
            return False
        
        # 超严格的流动性检查
        avg_volume_value = np.mean(closes[-20:] * volumes[-20:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 价格稳定性检查
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
        if volatility > 0.12:
            return False
        
        # 避免问题股票
        recent_decline = (closes[-1] / closes[-10] - 1)
        if recent_decline < -0.15:
            return False
        
        # 确保连续交易
        if closes[-1] == closes[-2] == closes[-3]:
            return False
        
        return True
    except:
        return False

def calculate_elite_quality_score(stock, context):
    """计算精英质量评分"""
    try:
        closes = history_bars(stock, 40, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 25, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 25:
            return 0
        
        score = 0
        
        # 价格趋势稳定性 (40%)
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma30 = np.mean(closes[-30:])
        
        if closes[-1] > ma10 > ma20 > ma30:
            score += 0.40
        elif closes[-1] > ma10 > ma20:
            score += 0.30
        elif closes[-1] > ma10:
            score += 0.20
        
        # 波动率控制 (30%)
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
        if volatility < 0.05:
            score += 0.30
        elif volatility < 0.08:
            score += 0.20
        elif volatility < 0.12:
            score += 0.10
        
        # RSI位置 (20%)
        rsi = calculate_smooth_rsi(closes, context.rsi_period)
        if 40 <= rsi <= 60:
            score += 0.20
        elif 30 <= rsi <= 70:
            score += 0.15
        elif 25 <= rsi <= 75:
            score += 0.10
        
        # 成交量稳定性 (10%)
        volume_stability = 1 - (np.std(volumes[-15:]) / np.mean(volumes[-15:]))
        score += 0.10 * min(volume_stability, 1.0)
        
        return min(score, 1.0)
    except:
        return 0

def calculate_elite_momentum_score(stock, context):
    """计算精英动量评分"""
    try:
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return 0
        
        score = 0
        
        # 短期强动量 (50%)
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        
        if momentum_3d > context.momentum_threshold:
            score += 0.25
        elif momentum_3d > 0.03:
            score += 0.20
        elif momentum_3d > 0.01:
            score += 0.15
        
        if momentum_5d > context.momentum_threshold * 1.5:
            score += 0.25
        elif momentum_5d > 0.04:
            score += 0.20
        elif momentum_5d > 0.02:
            score += 0.15
        
        # 加速度 (30%)
        if len(closes) >= 10:
            momentum_recent = (closes[-1] / closes[-4] - 1)
            momentum_prev = (closes[-4] / closes[-7] - 1)
            acceleration = momentum_recent - momentum_prev
            
            if acceleration > context.price_acceleration_min:
                score += 0.30
            elif acceleration > 0.01:
                score += 0.20
            elif acceleration > 0:
                score += 0.10
        
        # 相对强度 (20%)
        if len(closes) >= 15:
            recent_high = np.max(closes[-10:])
            if closes[-1] >= recent_high * 0.99:
                score += 0.20
            elif closes[-1] >= recent_high * 0.96:
                score += 0.15
            elif closes[-1] >= recent_high * 0.92:
                score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

def calculate_elite_volume_score(stock, context):
    """计算精英成交量评分"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return 0
        
        score = 0
        
        # 成交量突破 (50%)
        vol_recent = np.mean(volumes[-3:])
        vol_base = np.mean(volumes[-12:])
        vol_ratio = vol_recent / vol_base if vol_base > 0 else 1
        
        if vol_ratio > context.volume_multiplier:
            score += 0.50
        elif vol_ratio > 2.0:
            score += 0.40
        elif vol_ratio > 1.5:
            score += 0.30
        elif vol_ratio > 1.2:
            score += 0.20
        
        # 价量配合 (35%)
        price_change = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) >= 2 else 0
        
        if price_change > 0.02 and volume_change > 0.5:
            score += 0.35
        elif price_change > 0.01 and volume_change > 0.3:
            score += 0.25
        elif price_change > 0 and volume_change > 0:
            score += 0.15
        
        # 持续性 (15%)
        if len(volumes) >= 5:
            volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
            if volume_trend > 0:
                score += 0.15
            elif volume_trend > -volumes[-5] * 0.1:
                score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

def calculate_smooth_rsi(prices, period):
    """计算平滑RSI"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # 使用平滑移动平均
    alpha = 2.0 / (period + 1)
    avg_gain = gains[0]
    avg_loss = losses[0]
    
    for i in range(1, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_elite_position_size(context, score, position_index):
    """计算精英仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 评分加权 - 高分获得更大仓位
    score_multiplier = 0.8 + 0.6 * (score - 0.9) / 0.1  # 分数在0.9-1.0间的加权
    
    # 市场状态调整
    if context.market_state == 'bull':
        market_multiplier = 1.15
    elif context.market_state == 'neutral':
        market_multiplier = 1.05
    else:
        market_multiplier = 0.95
    
    # 位置优先级
    priority_multiplier = 1.5 if position_index == 0 else (1.2 if position_index == 1 else 1.0)
    
    # 计算最终仓位
    position_size = base_size * score_multiplier * market_multiplier * priority_multiplier
    
    # 限制最大仓位
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0)