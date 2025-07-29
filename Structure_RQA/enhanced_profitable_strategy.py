# -*- coding: utf-8 -*-
"""
Enhanced Profitable Strategy 2025 v2.0
=======================================
基于首次回测结果优化的高收益策略

优化要点：
1. 更严格的股票筛选 - 提高信号质量
2. 更激进的仓位管理 - 追求更高收益
3. 改进的风控机制 - 减少大额损失
4. 市场状态适应性 - 针对不同市场环境调整策略
5. 加强的技术指标 - 提高胜率
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
    # 核心参数 - 更激进设置
    context.max_positions = 6          # 减少至6只，集中火力
    context.base_position = 0.95       # 提高至95%仓位
    context.single_position_limit = 0.20  # 单股最大20%
    
    # 技术指标参数 - 更严格筛选
    context.volatility_window = 15     # 缩短波动率窗口
    context.volatility_threshold = 2.5 # 提高波动率阈值
    context.rsi_period = 10           # 缩短RSI周期，更敏感
    context.rsi_oversold = 20         # 更严格的超卖线
    context.rsi_overbought = 80       # 更严格的超买线
    
    # 突破参数 - 更高标准
    context.breakout_period = 8       # 缩短突破周期
    context.volume_multiplier = 2.2   # 提高成交量要求
    context.price_change_threshold = 0.04  # 提高价格变化阈值
    context.momentum_threshold = 0.05  # 新增动量阈值
    
    # 风控参数 - 更严格控制
    context.stop_loss = 0.05          # 降低至5%止损
    context.take_profit_fast = 0.10   # 快速10%止盈
    context.take_profit_medium = 0.18 # 中等18%止盈
    context.take_profit_high = 0.30   # 高收益30%止盈
    context.max_hold_days = 6         # 缩短至6天
    context.daily_loss_limit = 0.015  # 降低日损失限制
    
    # 筛选条件 - 更严格
    context.min_price = 10.0          # 提高最低价格
    context.max_price = 80.0          # 降低最高价格
    context.min_market_cap = 80e8     # 提高市值要求
    context.min_volume_value = 120e6  # 提高成交额要求
    context.min_avg_volume_days = 20  # 平均成交量天数
    
    # 市场状态参数
    context.market_volatility_period = 10
    context.bull_market_threshold = 0.015
    context.bear_market_threshold = -0.015
    
    # 新增质量评分参数
    context.quality_score_threshold = 0.8
    context.momentum_score_threshold = 0.75
    context.volume_score_threshold = 0.7
    
    # 状态记录
    context.positions = {}
    context.stock_universe = set()
    context.market_state = 'neutral'
    context.daily_returns = []
    context.performance_tracker = []
    context.scan_frequency = 1
    context.last_scan = None
    context.debug = True
    context.consecutive_losses = 0    # 连续亏损计数
    context.max_consecutive_losses = 3 # 最大连续亏损次数
    
    # 动态调整参数
    context.volatility_adjustment = 1.0
    context.position_adjustment = 1.0
    
    # 初始化股票池
    initialize_enhanced_universe(context)
    print(f"Enhanced Profitable Strategy v2.0 初始化完成 - 优质股票池: {len(context.stock_universe)}只")

def initialize_enhanced_universe(context):
    """初始化增强版股票池"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 排除特殊股票
        if any(x in symbol for x in ['ST', '*ST', '退', 'N', 'C']):
            continue
            
        # 只选择主板和深圳主板股票，排除高风险板块
        if stock.startswith('688') or stock.startswith('300'):
            continue
            
        # 排除新股 (简单过滤)
        if stock.startswith('301') and int(stock[3:6]) > 200:
            continue
            
        universe.add(stock)
    
    context.stock_universe = universe

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 更新市场状态
    update_enhanced_market_state(context)
    
    # 记录表现
    nav = context.portfolio.total_value
    daily_return = 0
    if context.performance_tracker:
        prev_nav = context.performance_tracker[-1]['nav']
        daily_return = (nav / prev_nav - 1)
    
    # 更新连续亏损计数
    if daily_return < -0.01:
        context.consecutive_losses += 1
    else:
        context.consecutive_losses = 0
    
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'daily_return': daily_return,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0]),
        'market_state': context.market_state,
        'consecutive_losses': context.consecutive_losses
    })
    
    # 动态调整策略参数
    adjust_strategy_parameters(context, daily_return)
    
    # 风控检查
    if daily_return < -context.daily_loss_limit or context.consecutive_losses >= context.max_consecutive_losses:
        if context.debug:
            print(f"{current_date} 触发风控限制，暂停新开仓")
        manage_enhanced_positions(context, bar_dict, current_date, risk_mode=True)
        return
    
    # 管理现有持仓
    manage_enhanced_positions(context, bar_dict, current_date)
    
    # 扫描新机会
    if should_scan_enhanced_opportunities(context, current_date):
        scan_enhanced_opportunities(context, bar_dict, current_date)

def adjust_strategy_parameters(context, daily_return):
    """动态调整策略参数"""
    # 根据近期表现调整激进程度
    if len(context.performance_tracker) >= 10:
        recent_returns = [p['daily_return'] for p in context.performance_tracker[-10:]]
        avg_return = np.mean(recent_returns)
        
        if avg_return > 0.01:  # 表现良好，更激进
            context.position_adjustment = 1.1
            context.volatility_adjustment = 0.9
        elif avg_return < -0.005:  # 表现不佳，更保守
            context.position_adjustment = 0.9
            context.volatility_adjustment = 1.1
        else:
            context.position_adjustment = 1.0
            context.volatility_adjustment = 1.0

def update_enhanced_market_state(context):
    """更新增强版市场状态"""
    try:
        benchmark_data = history_bars('000300.XSHG', 20, '1d', 'close', skip_suspended=True, include_now=True)
        
        if benchmark_data is None or len(benchmark_data) < 15:
            context.market_state = 'neutral'
            return
        
        # 短期趋势
        short_trend = (benchmark_data[-1] / benchmark_data[-5] - 1)
        # 中期趋势
        medium_trend = (benchmark_data[-1] / benchmark_data[-10] - 1)
        # 长期趋势
        long_trend = (benchmark_data[-1] / benchmark_data[-15] - 1)
        
        # 综合判断
        combined_trend = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
        
        if combined_trend > context.bull_market_threshold and short_trend > 0:
            context.market_state = 'bull'
        elif combined_trend < context.bear_market_threshold and short_trend < 0:
            context.market_state = 'bear'
        else:
            context.market_state = 'neutral'
            
    except:
        context.market_state = 'neutral'

def manage_enhanced_positions(context, bar_dict, current_date, risk_mode=False):
    """管理增强版持仓"""
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
        strategy_type = pos_info.get('strategy_type', 'breakout')
        profit_taking_level = pos_info.get('profit_taking_level', 0)
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        # 退出条件判断
        should_close = False
        should_reduce = False
        reason = ""
        
        if risk_mode:
            # 风控模式：快速止损
            if return_pct <= -context.stop_loss * 0.8:
                should_close, reason = True, f"风控止损，损失: {return_pct:.2%}"
        else:
            # 分级止盈策略
            if profit_taking_level == 0 and return_pct >= context.take_profit_fast:
                # 第一层止盈，减半
                should_reduce, reason = True, f"第一层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_taking_level'] = 1
            elif profit_taking_level == 1 and return_pct >= context.take_profit_medium:
                # 第二层止盈，再减半
                should_reduce, reason = True, f"第二层止盈减仓，收益: {return_pct:.2%}"
                pos_info['profit_taking_level'] = 2
            elif return_pct >= context.take_profit_high:
                # 第三层止盈，全部清仓
                should_close, reason = True, f"高收益止盈，收益: {return_pct:.2%}"
            elif return_pct <= -context.stop_loss:
                should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            elif hold_days >= context.max_hold_days:
                should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            elif check_enhanced_exit_signal(stock, context, strategy_type):
                should_close, reason = True, f"技术信号退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason))
        elif should_reduce:
            to_reduce.append((stock, reason))
    
    # 执行减仓
    for stock, reason in to_reduce:
        position = context.portfolio.positions.get(stock)
        current_weight = position.market_value / context.portfolio.total_value
        new_weight = current_weight * 0.5
        order_target_percent(stock, new_weight)
        if context.debug:
            print(f"{current_date} 减仓 {stock}: {reason}")
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_enhanced_exit_signal(stock, context, strategy_type):
    """检查增强版退出信号"""
    try:
        closes = history_bars(stock, 12, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 10, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 8:
            return False
        
        # 更严格的退出条件
        if strategy_type == 'breakout':
            # 连续2天下跌超过2%
            recent_changes = np.diff(closes[-3:]) / closes[-3:-1]
            if np.sum(recent_changes < -0.02) >= 2:
                return True
            
            # 成交量大幅萎缩
            vol_ratio = np.mean(volumes[-2:]) / np.mean(volumes[-6:])
            if vol_ratio < 0.3:
                return True
                
        elif strategy_type == 'mean_reversion':
            # RSI过度超买且开始下跌
            rsi = calculate_enhanced_rsi(closes, 8)
            if rsi > context.rsi_overbought and closes[-1] < closes[-2]:
                return True
        
        return False
    except:
        return False

def should_scan_enhanced_opportunities(context, current_date):
    """判断是否需要扫描增强版机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    # 在连续亏损时更谨慎
    if context.consecutive_losses >= 2:
        return False
    
    if current_positions < context.max_positions:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= 1
    else:
        return False

def scan_enhanced_opportunities(context, bar_dict, current_date):
    """扫描增强版机会"""
    if context.debug:
        print(f"{current_date} 扫描增强版机会 - 市场状态: {context.market_state}")
    
    # 根据市场状态调整扫描策略
    if context.market_state == 'bull':
        focus_on_breakout = True
        min_score_threshold = 0.85
    elif context.market_state == 'bear':
        focus_on_breakout = False
        min_score_threshold = 0.90
    else:
        focus_on_breakout = None
        min_score_threshold = 0.80
    
    candidates = []
    scan_stocks = list(context.stock_universe)[:200]  # 限制扫描范围
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            # 更严格的基础筛选
            if not enhanced_stock_filter(stock, bar_dict, context):
                continue
            
            # 综合评分系统
            quality_score = calculate_quality_score(stock, context)
            momentum_score = calculate_momentum_score(stock, context)
            volume_score = calculate_volume_score(stock, context)
            
            # 综合评分
            if focus_on_breakout is True:
                combined_score = momentum_score * 0.5 + quality_score * 0.3 + volume_score * 0.2
                strategy_type = 'breakout'
            elif focus_on_breakout is False:
                combined_score = quality_score * 0.5 + momentum_score * 0.2 + volume_score * 0.3
                strategy_type = 'mean_reversion'
            else:
                combined_score = (momentum_score + quality_score + volume_score) / 3
                strategy_type = 'breakout' if momentum_score > quality_score else 'mean_reversion'
            
            if combined_score >= min_score_threshold:
                candidates.append((stock, combined_score, strategy_type))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现优质机会")
        context.last_scan = current_date
        return
    
    # 按评分排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    for i, (stock, score, strategy_type) in enumerate(candidates[:available_slots]):
        position_size = calculate_enhanced_position_size(context, score, strategy_type, i)
        
        if position_size > 0.05:  # 最小5%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'strategy_type': strategy_type,
                'score': score,
                'profit_taking_level': 0
            }
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 策略: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.2f}")
    
    context.last_scan = current_date

def enhanced_stock_filter(stock, bar_dict, context):
    """增强版股票筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取更多历史数据
        closes = history_bars(stock, 40, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 30, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 25:
            return False
        
        # 严格的流动性检查
        avg_volume_value = np.mean(closes[-context.min_avg_volume_days:] * volumes[-context.min_avg_volume_days:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 避免问题股票
        recent_decline = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        if recent_decline < -0.12:
            return False
        
        # 避免异常波动
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
        if volatility > 0.15:  # 10日波动率超过15%
            return False
        
        # 确保不是停牌股
        if closes[-1] == closes[-2] == closes[-3]:
            return False
        
        return True
    
    except:
        return False

def calculate_quality_score(stock, context):
    """计算股票质量评分"""
    try:
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return 0
        
        score = 0
        
        # 价格稳定性 (30%)
        volatility = np.std(closes[-15:]) / np.mean(closes[-15:])
        if volatility < 0.08:
            score += 0.3
        elif volatility < 0.12:
            score += 0.2
        elif volatility < 0.15:
            score += 0.1
        
        # 趋势一致性 (30%)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        
        if closes[-1] > ma5 > ma10 > ma20:
            score += 0.3
        elif closes[-1] > ma5 > ma10:
            score += 0.2
        elif closes[-1] > ma5:
            score += 0.1
        
        # 成交量稳定性 (25%)
        volume_cv = np.std(volumes[-10:]) / np.mean(volumes[-10:])
        if volume_cv < 0.5:
            score += 0.25
        elif volume_cv < 0.8:
            score += 0.15
        elif volume_cv < 1.2:
            score += 0.1
        
        # RSI位置 (15%)
        rsi = calculate_enhanced_rsi(closes, context.rsi_period)
        if 30 <= rsi <= 70:
            score += 0.15
        elif 25 <= rsi <= 75:
            score += 0.1
        elif 20 <= rsi <= 80:
            score += 0.05
        
        return min(score, 1.0)
    except:
        return 0

def calculate_momentum_score(stock, context):
    """计算动量评分"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return 0
        
        score = 0
        current = closes[-1]
        
        # 多时间框架动量 (50%)
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        momentum_10d = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
        
        if momentum_3d > context.momentum_threshold:
            score += 0.2
        elif momentum_3d > 0.02:
            score += 0.15
        elif momentum_3d > 0:
            score += 0.1
        
        if momentum_5d > context.momentum_threshold * 1.5:
            score += 0.2
        elif momentum_5d > 0.03:
            score += 0.15
        elif momentum_5d > 0:
            score += 0.1
        
        if momentum_10d > 0.05:
            score += 0.1
        elif momentum_10d > 0:
            score += 0.05
        
        # 加速度 (25%)
        if len(closes) >= 8:
            accel = momentum_3d - (closes[-4] / closes[-7] - 1)
            if accel > 0.02:
                score += 0.25
            elif accel > 0:
                score += 0.15
        
        # 相对强度 (25%)
        if len(closes) >= 20:
            recent_high = np.max(closes[-10:])
            if current >= recent_high * 0.98:
                score += 0.25
            elif current >= recent_high * 0.95:
                score += 0.15
            elif current >= recent_high * 0.90:
                score += 0.1
        
        return min(score, 1.0)
    except:
        return 0

def calculate_volume_score(stock, context):
    """计算成交量评分"""
    try:
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return 0
        
        score = 0
        
        # 成交量放大 (40%)
        vol_recent = np.mean(volumes[-3:])
        vol_base = np.mean(volumes[-10:])
        vol_ratio = vol_recent / vol_base if vol_base > 0 else 1
        
        if vol_ratio > context.volume_multiplier:
            score += 0.4
        elif vol_ratio > 1.8:
            score += 0.3
        elif vol_ratio > 1.3:
            score += 0.2
        elif vol_ratio > 1.0:
            score += 0.1
        
        # 价量配合 (30%)
        price_change = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) >= 2 else 0
        
        if price_change > 0 and volume_change > 0:
            if price_change > 0.03 and volume_change > 0.5:
                score += 0.3
            elif price_change > 0.01 and volume_change > 0.2:
                score += 0.2
            else:
                score += 0.1
        
        # 成交量趋势 (30%)
        if len(volumes) >= 10:
            volume_trend = np.polyfit(range(10), volumes[-10:], 1)[0]
            if volume_trend > 0:
                score += 0.3
            elif volume_trend > -volumes[-10] * 0.1:
                score += 0.15
        
        return min(score, 1.0)
    except:
        return 0

def calculate_enhanced_rsi(prices, period):
    """计算增强版RSI"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # 使用EMA而不是简单平均
    alpha = 1.0 / period
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

def calculate_enhanced_position_size(context, score, strategy_type, position_index):
    """计算增强版仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 根据评分大幅调整
    score_multiplier = 0.7 + 0.8 * score  # 更大的评分影响
    
    # 根据策略类型和市场状态调整
    if context.market_state == 'bull' and strategy_type == 'breakout':
        strategy_multiplier = 1.3
    elif context.market_state == 'bear' and strategy_type == 'mean_reversion':
        strategy_multiplier = 1.2
    elif context.market_state == 'neutral':
        strategy_multiplier = 1.1
    else:
        strategy_multiplier = 1.0
    
    # 首选股票获得显著更大仓位
    if position_index == 0:
        priority_multiplier = 1.4
    elif position_index == 1:
        priority_multiplier = 1.2
    else:
        priority_multiplier = 1.0
    
    # 应用动态调整
    position_size = base_size * score_multiplier * strategy_multiplier * priority_multiplier * context.position_adjustment
    
    # 限制单股最大仓位
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0)