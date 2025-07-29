# -*- coding: utf-8 -*-
"""
Optimized Profitable Strategy 2025 v4.0
=======================================
基于Enhanced Strategy成功经验的优化版本
平衡了信号质量和交易机会频率

核心改进：
1. 降低过严的准入门槛，保持质量导向
2. 保留成功的分层止盈机制
3. 优化仓位管理和资金使用效率
4. 提高交易频率，增加盈利机会
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
    # 核心参数 - 平衡配置
    context.max_positions = 5          # 5只股票平衡风险和集中度
    context.base_position = 0.93       # 93%仓位，保留缓冲
    context.single_position_limit = 0.25  # 单股最大25%
    
    # 技术指标参数 - 适中标准
    context.volatility_window = 15
    context.volatility_threshold = 2.3
    context.rsi_period = 12
    context.rsi_oversold = 25
    context.rsi_overbought = 75
    
    # 突破参数 - 平衡的信号标准
    context.breakout_period = 8
    context.volume_multiplier = 2.0
    context.momentum_threshold = 0.04
    context.price_acceleration_min = 0.02
    
    # 风控参数 - 经验证的设置
    context.stop_loss = 0.05          # 5%止损
    context.take_profit_layer1 = 0.10 # 第一层10%止盈
    context.take_profit_layer2 = 0.18 # 第二层18%止盈  
    context.take_profit_layer3 = 0.30 # 第三层30%止盈
    context.max_hold_days = 8         # 最大持有8天
    context.daily_loss_limit = 0.015  # 1.5%日损失限制
    
    # 筛选条件 - 合理门槛
    context.min_price = 8.0
    context.max_price = 100.0
    context.min_market_cap = 60e8     # 60亿市值
    context.min_volume_value = 100e6  # 1亿成交额
    
    # 评分阈值 - 可达到的标准
    context.min_total_score = 0.75    # 降低总评分要求
    context.min_quality_score = 0.65  # 降低质量要求
    context.min_momentum_score = 0.70 # 保持动量要求
    context.min_volume_score = 0.60   # 降低成交量要求
    
    # 市场状态参数
    context.bull_threshold = 0.012
    context.bear_threshold = -0.012
    
    # 状态记录
    context.positions = {}
    context.stock_universe = set()
    context.market_state = 'neutral'
    context.performance_tracker = []
    context.scan_frequency = 1        # 每天扫描
    context.last_scan = None
    context.debug = True
    context.total_trades = 0
    context.winning_trades = 0
    context.consecutive_losses = 0
    
    # 初始化股票池
    initialize_balanced_universe(context)
    print(f"Optimized Profitable Strategy v4.0 初始化完成 - 股票池: {len(context.stock_universe)}只")

def initialize_balanced_universe(context):
    """初始化平衡股票池"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 排除问题股票
        if any(x in symbol for x in ['ST', '*ST', '退', 'N']):
            continue
            
        # 适度限制高风险板块，但不完全排除
        if stock.startswith('688'):  # 科创板要求更高
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
    
    # 更新连续亏损计数
    if daily_return < -0.008:
        context.consecutive_losses += 1
    else:
        context.consecutive_losses = 0
    
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'daily_return': daily_return,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0]),
        'market_state': context.market_state,
        'total_trades': context.total_trades,
        'win_rate': context.winning_trades / max(context.total_trades, 1) * 100,
        'consecutive_losses': context.consecutive_losses
    })
    
    # 风控检查
    risk_mode = (daily_return < -context.daily_loss_limit or 
                context.consecutive_losses >= 3)
    
    if risk_mode and context.debug:
        print(f"{current_date} 触发风控，限制新开仓")
    
    # 管理现有持仓
    manage_optimized_positions(context, bar_dict, current_date, risk_mode)
    
    # 扫描新机会
    if not risk_mode and should_scan_opportunities(context, current_date):
        scan_optimized_opportunities(context, bar_dict, current_date)

def update_market_state(context):
    """更新市场状态"""
    try:
        benchmark_data = history_bars('000300.XSHG', 12, '1d', 'close', skip_suspended=True, include_now=True)
        
        if benchmark_data is None or len(benchmark_data) < 8:
            context.market_state = 'neutral'
            return
        
        # 短中期趋势结合
        short_trend = (benchmark_data[-1] / benchmark_data[-4] - 1)
        medium_trend = (benchmark_data[-1] / benchmark_data[-8] - 1)
        combined_trend = short_trend * 0.6 + medium_trend * 0.4
        
        if combined_trend > context.bull_threshold:
            context.market_state = 'bull'
        elif combined_trend < context.bear_threshold:
            context.market_state = 'bear'
        else:
            context.market_state = 'neutral'
            
    except:
        context.market_state = 'neutral'

def manage_optimized_positions(context, bar_dict, current_date, risk_mode=False):
    """管理优化持仓"""
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
        
        should_close = False
        should_reduce = False
        reason = ""
        
        if risk_mode:
            # 风控模式：快速止损
            if return_pct <= -context.stop_loss * 0.8:
                should_close, reason = True, f"风控止损，损失: {return_pct:.2%}"
        else:
            # 正常的分层止盈止损
            if profit_level == 0 and return_pct >= context.take_profit_layer1:
                should_reduce, reason = True, f"一层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 1
            elif profit_level == 1 and return_pct >= context.take_profit_layer2:
                should_reduce, reason = True, f"二层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_level'] = 2
            elif return_pct >= context.take_profit_layer3:
                should_close, reason = True, f"三层止盈全清，收益: {return_pct:.2%}"
            elif return_pct <= -context.stop_loss:
                should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            elif hold_days >= context.max_hold_days:
                should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            elif check_technical_exit(stock, context):
                should_close, reason = True, f"技术退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason, return_pct))
        elif should_reduce:
            to_reduce.append((stock, reason))
    
    # 执行减仓 - 减至原仓位的60%
    for stock, reason in to_reduce:
        position = context.portfolio.positions.get(stock)
        current_weight = position.market_value / context.portfolio.total_value
        new_weight = current_weight * 0.6
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

def check_technical_exit(stock, context):
    """检查技术退出信号"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 6, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 6:
            return False
        
        # 连续下跌信号
        if len(closes) >= 4:
            recent_decline = (closes[-1] / closes[-4] - 1)
            if recent_decline < -0.04:  # 3天跌4%以上
                return True
        
        # 成交量萎缩
        if len(volumes) >= 5:
            vol_ratio = np.mean(volumes[-2:]) / np.mean(volumes[-5:])
            if vol_ratio < 0.4:  # 成交量萎缩60%以上
                return True
        
        return False
    except:
        return False

def should_scan_opportunities(context, current_date):
    """判断是否扫描机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    if current_positions >= context.max_positions:
        return False
    
    if context.last_scan is None:
        return True
        
    days_since_scan = (current_date - context.last_scan).days
    return days_since_scan >= context.scan_frequency

def scan_optimized_opportunities(context, bar_dict, current_date):
    """扫描优化机会"""
    if context.debug:
        print(f"{current_date} 扫描机会 - 市场: {context.market_state}")
    
    candidates = []
    scan_stocks = list(context.stock_universe)[:250]  # 适中的扫描范围
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            if not balanced_stock_filter(stock, bar_dict, context):
                continue
            
            # 三维评分
            quality_score = calculate_balanced_quality_score(stock, context)
            momentum_score = calculate_balanced_momentum_score(stock, context)
            volume_score = calculate_balanced_volume_score(stock, context)
            
            # 检查各维度是否达标
            if (quality_score >= context.min_quality_score and 
                momentum_score >= context.min_momentum_score and 
                volume_score >= context.min_volume_score):
                
                # 根据市场状态调整权重
                if context.market_state == 'bull':
                    total_score = momentum_score * 0.5 + quality_score * 0.3 + volume_score * 0.2
                    strategy_type = 'breakout'
                elif context.market_state == 'bear':
                    total_score = quality_score * 0.5 + volume_score * 0.3 + momentum_score * 0.2
                    strategy_type = 'quality'
                else:
                    total_score = (quality_score + momentum_score + volume_score) / 3
                    strategy_type = 'balanced'
                
                if total_score >= context.min_total_score:
                    candidates.append((stock, total_score, strategy_type))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现合格机会")
        context.last_scan = current_date
        return
    
    # 按评分排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    for i, (stock, score, strategy_type) in enumerate(candidates[:available_slots]):
        position_size = calculate_balanced_position_size(context, score, i)
        
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
                print(f"{current_date} 开仓 {stock}, 类型: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.2f}")
    
    context.last_scan = current_date

def balanced_stock_filter(stock, bar_dict, context):
    """平衡股票筛选"""
    try:
        price = bar_dict[stock].close
        
        if not (context.min_price <= price <= context.max_price):
            return False
        
        closes = history_bars(stock, 35, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 25, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-15:] * volumes[-15:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 避免极端情况
        recent_decline = (closes[-1] / closes[-7] - 1)
        if recent_decline < -0.20:  # 一周跌超20%
            return False
        
        # 波动率检查
        volatility = np.std(closes[-15:]) / np.mean(closes[-15:])
        if volatility > 0.18:  # 波动率过高
            return False
        
        # 确保正常交易
        if closes[-1] == closes[-2] == closes[-3]:
            return False
        
        return True
    except:
        return False

def calculate_balanced_quality_score(stock, context):
    """计算平衡质量评分"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 18, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 18:
            return 0
        
        score = 0
        
        # 趋势质量 (40%)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma15 = np.mean(closes[-15:])
        
        if closes[-1] > ma5 > ma10 > ma15:
            score += 0.40
        elif closes[-1] > ma5 > ma10:
            score += 0.30
        elif closes[-1] > ma5:
            score += 0.20
        elif closes[-1] > ma10:
            score += 0.10
        
        # 稳定性 (30%)
        volatility = np.std(closes[-12:]) / np.mean(closes[-12:])
        if volatility < 0.06:
            score += 0.30
        elif volatility < 0.10:
            score += 0.20
        elif volatility < 0.15:
            score += 0.10
        
        # RSI合理性 (20%)
        rsi = calculate_rsi(closes, context.rsi_period)
        if 35 <= rsi <= 65:
            score += 0.20
        elif 25 <= rsi <= 75:
            score += 0.15
        elif 20 <= rsi <= 80:
            score += 0.10
        
        # 价格位置 (10%)
        recent_high = np.max(closes[-15:])
        price_position = closes[-1] / recent_high
        if price_position > 0.90:
            score += 0.10
        elif price_position > 0.80:
            score += 0.08
        elif price_position > 0.70:
            score += 0.05
        
        return min(score, 1.0)
    except:
        return 0

def calculate_balanced_momentum_score(stock, context):
    """计算平衡动量评分"""
    try:
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        
        if closes is None or len(closes) < 15:
            return 0
        
        score = 0
        
        # 短期动量 (50%)
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        
        if momentum_3d > context.momentum_threshold:
            score += 0.25
        elif momentum_3d > 0.02:
            score += 0.20
        elif momentum_3d > 0.01:
            score += 0.15
        elif momentum_3d > 0:
            score += 0.10
        
        if momentum_5d > context.momentum_threshold * 1.2:
            score += 0.25
        elif momentum_5d > 0.03:
            score += 0.20
        elif momentum_5d > 0.01:
            score += 0.15
        elif momentum_5d > 0:
            score += 0.10
        
        # 加速度 (25%)
        if len(closes) >= 8:
            accel = momentum_3d - (closes[-4] / closes[-7] - 1)
            if accel > context.price_acceleration_min:
                score += 0.25
            elif accel > 0.01:
                score += 0.15
            elif accel > 0:
                score += 0.10
        
        # 相对强度 (25%)
        if len(closes) >= 12:
            recent_high = np.max(closes[-8:])
            if closes[-1] >= recent_high * 0.98:
                score += 0.25
            elif closes[-1] >= recent_high * 0.95:
                score += 0.20
            elif closes[-1] >= recent_high * 0.90:
                score += 0.15
            elif closes[-1] >= recent_high * 0.85:
                score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

def calculate_balanced_volume_score(stock, context):
    """计算平衡成交量评分"""
    try:
        closes = history_bars(stock, 18, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(volumes) < 12:
            return 0
        
        score = 0
        
        # 成交量放大 (45%)
        vol_recent = np.mean(volumes[-3:])
        vol_base = np.mean(volumes[-10:])
        vol_ratio = vol_recent / vol_base if vol_base > 0 else 1
        
        if vol_ratio > context.volume_multiplier:
            score += 0.45
        elif vol_ratio > 1.7:
            score += 0.35
        elif vol_ratio > 1.4:
            score += 0.25
        elif vol_ratio > 1.1:
            score += 0.15
        elif vol_ratio >= 1.0:
            score += 0.10
        
        # 价量配合 (35%)
        price_change = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) >= 2 else 0
        
        if price_change > 0.015 and volume_change > 0.3:
            score += 0.35
        elif price_change > 0.008 and volume_change > 0.1:
            score += 0.25
        elif price_change > 0 and volume_change > 0:
            score += 0.15
        elif price_change > 0:
            score += 0.10
        
        # 持续性 (20%)
        if len(volumes) >= 6:
            vol_trend = np.polyfit(range(6), volumes[-6:], 1)[0]
            if vol_trend > 0:
                score += 0.20
            elif vol_trend > -volumes[-6] * 0.1:
                score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

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

def calculate_balanced_position_size(context, score, position_index):
    """计算平衡仓位大小"""
    base_size = context.base_position / context.max_positions
    
    # 评分权重
    score_multiplier = 0.8 + 0.4 * (score - 0.75) / 0.25
    
    # 市场状态调整
    if context.market_state == 'bull':
        market_multiplier = 1.10
    elif context.market_state == 'neutral':
        market_multiplier = 1.05
    else:
        market_multiplier = 1.00
    
    # 优先级调整
    priority_multiplier = 1.3 if position_index == 0 else (1.15 if position_index == 1 else 1.0)
    
    position_size = base_size * score_multiplier * market_multiplier * priority_multiplier
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0)