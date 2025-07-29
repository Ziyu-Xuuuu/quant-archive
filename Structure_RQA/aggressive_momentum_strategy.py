# -*- coding: utf-8 -*-
"""
Aggressive Momentum Strategy 2025 v1.0
======================================
专为实现25%+年化收益率设计的激进动量策略

核心策略：
1. 超短线高频交易 - 3-5天持有期
2. 极端动量捕捉 - 寻找爆发性机会
3. 激进仓位管理 - 高杠杆式集中持仓
4. 快速止盈止损 - 锁定收益，控制风险
5. 多策略融合 - 突破+反转+趋势跟踪
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
    # 激进配置参数
    context.max_positions = 3          # 极度集中持仓3只
    context.base_position = 0.98       # 几乎满仓
    context.single_position_limit = 0.40  # 单股最大40%
    
    # 激进技术参数
    context.momentum_threshold = 0.08   # 8%动量阈值
    context.volume_surge_min = 3.0      # 成交量至少放大3倍
    context.volatility_breakout = 3.0   # 波动率突破3倍
    context.price_acceleration = 0.05   # 5%价格加速度
    
    # 超短线风控
    context.stop_loss = 0.06           # 6%快速止损
    context.profit_target_1 = 0.15     # 15%第一目标
    context.profit_target_2 = 0.25     # 25%第二目标
    context.profit_target_3 = 0.40     # 40%终极目标
    context.max_hold_days = 5          # 最大5天持有
    context.intraday_loss_limit = 0.03 # 单日3%损失限制
    
    # 筛选条件
    context.min_price = 5.0
    context.max_price = 200.0
    context.min_volume_value = 200e6   # 2亿成交额
    context.min_market_cap = 50e8      # 50亿市值
    
    # 评分标准 - 激进设置
    context.min_momentum_score = 0.85  # 极高动量要求
    context.min_volume_score = 0.80    # 极高成交量要求
    context.min_total_score = 0.90     # 极高综合评分
    
    # 策略状态
    context.positions = {}
    context.stock_universe = set()
    context.market_regime = 'normal'
    context.daily_pnl_tracker = []
    context.trade_stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'total_return': 0,
        'max_single_return': 0,
        'max_single_loss': 0
    }
    context.debug = True
    context.high_momentum_pool = []
    context.last_scan = None
    
    # 初始化股票池
    initialize_aggressive_universe(context)
    print(f"Aggressive Momentum Strategy v1.0 初始化完成 - 激进股票池: {len(context.stock_universe)}只")

def initialize_aggressive_universe(context):
    """初始化激进股票池 - 包含所有活跃股票"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 只排除明显问题股票
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
            
        universe.add(stock)
    
    context.stock_universe = universe

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 更新市场状态
    update_market_regime(context)
    
    # 记录当日表现
    nav = context.portfolio.total_value
    daily_return = 0
    if context.daily_pnl_tracker:
        prev_nav = context.daily_pnl_tracker[-1]['nav']
        daily_return = (nav / prev_nav - 1)
    
    context.daily_pnl_tracker.append({
        'date': current_date,
        'nav': nav,
        'daily_return': daily_return,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0]),
        'regime': context.market_regime
    })
    
    # 计算年化收益率
    if len(context.daily_pnl_tracker) > 20:
        total_return = (nav / 1_000_000 - 1)
        days_elapsed = len(context.daily_pnl_tracker)
        annualized_return = ((1 + total_return) ** (252/days_elapsed) - 1) * 100
        
        if context.debug and days_elapsed % 20 == 0:
            print(f"{current_date} 当前年化收益率: {annualized_return:.1f}%, 总收益: {total_return:.2%}")
    
    # 风控检查
    risk_mode = daily_return < -context.intraday_loss_limit
    if risk_mode and context.debug:
        print(f"{current_date} 触发风控，当日损失: {daily_return:.2%}")
    
    # 管理现有持仓
    manage_aggressive_positions(context, bar_dict, current_date, risk_mode)
    
    # 每日扫描机会
    if not risk_mode:
        scan_explosive_opportunities(context, bar_dict, current_date)

def update_market_regime(context):
    """更新市场状态"""
    try:
        benchmark = history_bars('000300.XSHG', 10, '1d', 'close', skip_suspended=True, include_now=True)
        if benchmark is None or len(benchmark) < 8:
            context.market_regime = 'normal'
            return
        
        short_momentum = (benchmark[-1] / benchmark[-3] - 1)
        medium_momentum = (benchmark[-1] / benchmark[-6] - 1)
        
        if short_momentum > 0.02 and medium_momentum > 0.03:
            context.market_regime = 'bull_explosive'
        elif short_momentum < -0.02 and medium_momentum < -0.03:
            context.market_regime = 'bear_crash'
        elif abs(short_momentum) > 0.015:
            context.market_regime = 'volatile'
        else:
            context.market_regime = 'normal'
            
    except:
        context.market_regime = 'normal'

def manage_aggressive_positions(context, bar_dict, current_date, risk_mode=False):
    """激进持仓管理"""
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
        strategy_type = pos_info.get('strategy_type', 'momentum')
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        should_close = False
        should_reduce = False
        reason = ""
        
        if risk_mode:
            # 风控模式快速止损
            if return_pct <= -context.stop_loss * 0.7:
                should_close, reason = True, f"风控止损，损失: {return_pct:.2%}"
        else:
            # 分层快速止盈
            if profit_level == 0 and return_pct >= context.profit_target_1:
                should_reduce, reason = True, f"一层止盈({context.profit_target_1:.0%})，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 1
            elif profit_level == 1 and return_pct >= context.profit_target_2:
                should_reduce, reason = True, f"二层止盈({context.profit_target_2:.0%})，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 2
            elif return_pct >= context.profit_target_3:
                should_close, reason = True, f"终极止盈({context.profit_target_3:.0%})，收益: {return_pct:.2%}"
            elif return_pct <= -context.stop_loss:
                should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            elif hold_days >= context.max_hold_days:
                should_close, reason = True, f"超短线时间止损，收益: {return_pct:.2%}"
            elif check_momentum_reversal(stock, context):
                should_close, reason = True, f"动量反转退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason, return_pct))
        elif should_reduce:
            to_reduce.append((stock, reason))
    
    # 执行减仓 - 激进减至50%
    for stock, reason in to_reduce:
        position = context.portfolio.positions.get(stock)
        current_weight = position.market_value / context.portfolio.total_value
        new_weight = current_weight * 0.5
        order_target_percent(stock, new_weight)
        if context.debug:
            print(f"{current_date} 减仓 {stock}: {reason}")
    
    # 执行平仓并更新统计
    for stock, reason, return_pct in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        
        # 更新交易统计
        context.trade_stats['total_trades'] += 1
        context.trade_stats['total_return'] += return_pct
        if return_pct > 0:
            context.trade_stats['winning_trades'] += 1
            context.trade_stats['max_single_return'] = max(context.trade_stats['max_single_return'], return_pct)
        else:
            context.trade_stats['max_single_loss'] = min(context.trade_stats['max_single_loss'], return_pct)
        
        if context.debug:
            win_rate = context.trade_stats['winning_trades'] / context.trade_stats['total_trades'] * 100
            avg_return = context.trade_stats['total_return'] / context.trade_stats['total_trades'] * 100
            print(f"{current_date} 平仓 {stock}: {reason}")
            print(f"  交易统计: 胜率{win_rate:.1f}%, 平均收益{avg_return:.2f}%, 最大单次{context.trade_stats['max_single_return']:.2%}")

def check_momentum_reversal(stock, context):
    """检查动量反转信号"""
    try:
        closes = history_bars(stock, 6, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 6, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 5:
            return False
        
        # 连续2天下跌且成交量萎缩
        price_trend = np.diff(closes[-3:])
        if np.sum(price_trend < 0) >= 2:
            vol_ratio = volumes[-1] / np.mean(volumes[-5:])
            if vol_ratio < 0.6:  # 成交量萎缩40%以上
                return True
        
        # 急剧放量下跌
        daily_change = (closes[-1] / closes[-2] - 1)
        volume_surge = volumes[-1] / np.mean(volumes[-4:-1])
        if daily_change < -0.03 and volume_surge > 2.0:
            return True
        
        return False
    except:
        return False

def scan_explosive_opportunities(context, bar_dict, current_date):
    """扫描爆发性机会"""
    if context.debug:
        print(f"{current_date} 扫描爆发性机会 - 市场状态: {context.market_regime}")
    
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    if current_positions >= context.max_positions:
        return
    
    candidates = []
    scan_limit = 300  # 扫描范围
    scan_stocks = list(context.stock_universe)[:scan_limit]
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            if not explosive_filter(stock, bar_dict, context):
                continue
            
            # 计算爆发性评分
            momentum_score = calculate_explosive_momentum(stock, context)
            volume_score = calculate_explosive_volume(stock, context)
            
            if (momentum_score >= context.min_momentum_score and 
                volume_score >= context.min_volume_score):
                
                total_score = (momentum_score + volume_score) / 2
                
                if total_score >= context.min_total_score:
                    strategy_type = determine_strategy_type(stock, context)
                    candidates.append((stock, total_score, strategy_type))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现爆发性机会")
        return
    
    # 按评分排序，选择最佳
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 激进开仓
    available_slots = context.max_positions - current_positions
    for i, (stock, score, strategy_type) in enumerate(candidates[:available_slots]):
        position_size = calculate_aggressive_position_size(context, score, i)
        
        if position_size > 0.15:  # 最小15%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'strategy_type': strategy_type,
                'score': score,
                'profit_level': 0
            }
            
            if context.debug:
                print(f"{current_date} 激进开仓 {stock}, 策略: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.2f}")

def explosive_filter(stock, bar_dict, context):
    """爆发性机会筛选器"""
    try:
        price = bar_dict[stock].close
        
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取数据
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        highs = history_bars(stock, 10, '1d', 'high', skip_suspended=True, include_now=True)
        lows = history_bars(stock, 10, '1d', 'low', skip_suspended=True, include_now=True)
        
        if any(x is None for x in [closes, volumes, highs, lows]) or len(closes) < 15:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-5:] * volumes[-5:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 必须有明显的价格变动
        daily_change = abs(closes[-1] / closes[-2] - 1)
        if daily_change < 0.02:  # 至少2%的日变动
            return False
        
        # 避免连续大跌股票
        decline_check = (closes[-1] / closes[-5] - 1)
        if decline_check < -0.15:
            return False
        
        # 确保正常交易
        if closes[-1] == closes[-2]:
            return False
        
        return True
    except:
        return False

def calculate_explosive_momentum(stock, context):
    """计算爆发性动量评分"""
    try:
        closes = history_bars(stock, 15, '1d', 'close', skip_suspended=True, include_now=True)
        if closes is None or len(closes) < 10:
            return 0
        
        score = 0
        current = closes[-1]
        
        # 超短期爆发 (60%)
        momentum_1d = (closes[-1] / closes[-2] - 1)
        momentum_2d = (closes[-1] / closes[-3] - 1)
        momentum_3d = (closes[-1] / closes[-4] - 1)
        
        if momentum_1d > context.momentum_threshold:
            score += 0.25
        elif momentum_1d > 0.05:
            score += 0.20
        elif momentum_1d > 0.03:
            score += 0.15
        elif momentum_1d > 0.01:
            score += 0.10
        
        if momentum_2d > context.momentum_threshold * 1.2:
            score += 0.20
        elif momentum_2d > 0.06:
            score += 0.15
        elif momentum_2d > 0.03:
            score += 0.10
        
        if momentum_3d > context.momentum_threshold * 1.5:
            score += 0.15
        elif momentum_3d > 0.08:
            score += 0.10
        elif momentum_3d > 0.04:
            score += 0.05
        
        # 价格加速度 (25%)
        if len(closes) >= 6:
            accel = momentum_2d - (closes[-3] / closes[-5] - 1)
            if accel > context.price_acceleration:
                score += 0.25
            elif accel > 0.03:
                score += 0.15
            elif accel > 0.01:
                score += 0.10
        
        # 突破新高 (15%)
        recent_high = np.max(closes[-8:])
        if current >= recent_high:
            score += 0.15
        elif current >= recent_high * 0.98:
            score += 0.10
        elif current >= recent_high * 0.95:
            score += 0.05
        
        return min(score, 1.0)
    except:
        return 0

def calculate_explosive_volume(stock, context):
    """计算爆发性成交量评分"""
    try:
        closes = history_bars(stock, 12, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 12, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(volumes) < 8:
            return 0
        
        score = 0
        
        # 成交量爆发 (70%)
        current_volume = volumes[-1]
        avg_volume_5d = np.mean(volumes[-6:-1])
        avg_volume_10d = np.mean(volumes[-11:-1])
        
        surge_ratio_5d = current_volume / avg_volume_5d if avg_volume_5d > 0 else 1
        surge_ratio_10d = current_volume / avg_volume_10d if avg_volume_10d > 0 else 1
        
        if surge_ratio_5d > context.volume_surge_min * 2:
            score += 0.35
        elif surge_ratio_5d > context.volume_surge_min:
            score += 0.30
        elif surge_ratio_5d > 2.0:
            score += 0.25
        elif surge_ratio_5d > 1.5:
            score += 0.15
        elif surge_ratio_5d > 1.2:
            score += 0.10
        
        if surge_ratio_10d > context.volume_surge_min:
            score += 0.35
        elif surge_ratio_10d > 2.0:
            score += 0.25
        elif surge_ratio_10d > 1.5:
            score += 0.15
        
        # 价量配合 (30%)
        price_change = (closes[-1] / closes[-2] - 1)
        volume_change = (volumes[-1] / volumes[-2] - 1) if volumes[-2] > 0 else 0
        
        if price_change > 0.05 and volume_change > 2.0:
            score += 0.30
        elif price_change > 0.03 and volume_change > 1.0:
            score += 0.20
        elif price_change > 0.01 and volume_change > 0.5:
            score += 0.15
        elif price_change > 0 and volume_change > 0:
            score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

def determine_strategy_type(stock, context):
    """确定策略类型"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        if closes is None or len(closes) < 6:
            return 'momentum'
        
        # 基于近期表现确定策略
        recent_momentum = (closes[-1] / closes[-4] - 1)
        volatility = np.std(closes[-6:]) / np.mean(closes[-6:])
        
        if recent_momentum > 0.10 and volatility > 0.08:
            return 'explosive_breakout'
        elif recent_momentum > 0.05:
            return 'momentum_follow'
        elif recent_momentum < -0.03:
            return 'mean_reversion'
        else:
            return 'momentum'
    except:
        return 'momentum'

def calculate_aggressive_position_size(context, score, position_index):
    """计算激进仓位大小"""
    # 极度集中的仓位分配
    base_size = context.base_position / context.max_positions
    
    # 评分权重放大
    score_multiplier = 1.0 + (score - 0.9) * 2  # 高分获得巨大权重
    
    # 市场状态权重
    if context.market_regime == 'bull_explosive':
        regime_multiplier = 1.3
    elif context.market_regime == 'volatile':
        regime_multiplier = 1.2
    elif context.market_regime == 'bear_crash':
        regime_multiplier = 0.8
    else:
        regime_multiplier = 1.0
    
    # 优先级权重 - 极度倾斜
    if position_index == 0:
        priority_multiplier = 1.5
    elif position_index == 1:
        priority_multiplier = 1.2
    else:
        priority_multiplier = 1.0
    
    position_size = base_size * score_multiplier * regime_multiplier * priority_multiplier
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0.15)  # 最小15%