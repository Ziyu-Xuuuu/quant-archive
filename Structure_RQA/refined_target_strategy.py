# -*- coding: utf-8 -*-
"""
Refined Target Strategy 2025 v1.0
=================================
基于激进策略经验优化，专注实现稳定25%+年化收益

核心改进：
1. 保留高动量捕捉能力，降低过度激进风险
2. 更精确的止盈止损机制
3. 动态仓位管理，提高资金使用效率
4. 增强风控，减少大额损失
5. 优化选股标准，提高胜率
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
    # 平衡激进配置
    context.max_positions = 4          # 4只股票集中持仓
    context.base_position = 0.95       # 95%高仓位
    context.single_position_limit = 0.35  # 单股最大35%
    
    # 优化的技术参数
    context.momentum_threshold = 0.06   # 6%动量阈值
    context.volume_surge_min = 2.5      # 成交量放大2.5倍
    context.volatility_breakout = 2.5   # 波动率突破
    context.price_acceleration = 0.04   # 4%价格加速度
    
    # 精确的风控参数
    context.stop_loss = 0.07           # 7%止损
    context.profit_target_1 = 0.12     # 12%第一目标
    context.profit_target_2 = 0.22     # 22%第二目标
    context.profit_target_3 = 0.35     # 35%终极目标
    context.max_hold_days = 6          # 最大6天持有
    context.daily_loss_limit = 0.025   # 2.5%日损失限制
    
    # 筛选条件优化
    context.min_price = 6.0
    context.max_price = 150.0
    context.min_volume_value = 150e6   # 1.5亿成交额
    context.min_market_cap = 40e8      # 40亿市值
    
    # 评分标准调整
    context.min_momentum_score = 0.80  # 高动量要求
    context.min_volume_score = 0.75    # 高成交量要求
    context.min_quality_score = 0.70   # 适中质量要求
    context.min_total_score = 0.85     # 高综合评分
    
    # 策略状态
    context.positions = {}
    context.stock_universe = set()
    context.market_regime = 'normal'
    context.performance_tracker = []
    context.trade_stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'total_return': 0,
        'max_single_return': 0,
        'max_single_loss': 0,
        'consecutive_losses': 0
    }
    context.debug = True
    context.target_return = 25.0  # 目标年化收益率25%
    
    # 初始化股票池
    initialize_refined_universe(context)
    print(f"Refined Target Strategy v1.0 初始化完成 - 目标年化收益: {context.target_return}%")
    print(f"优化股票池: {len(context.stock_universe)}只")

def initialize_refined_universe(context):
    """初始化优化股票池"""
    instruments = all_instruments('CS')
    universe = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 排除问题股票
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
        
        # 适度限制高风险板块
        if stock.startswith('688'):  # 排除科创板
            continue
            
        universe.add(stock)
    
    context.stock_universe = universe

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 更新市场状态
    update_market_regime(context)
    
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
        'regime': context.market_regime
    })
    
    # 计算当前年化收益率
    if len(context.performance_tracker) > 10:
        total_return = (nav / 1_000_000 - 1)
        days_elapsed = len(context.performance_tracker)
        annualized_return = ((1 + total_return) ** (252/days_elapsed) - 1) * 100
        
        # 定期报告进度
        if days_elapsed % 15 == 0:
            win_rate = context.trade_stats['winning_trades'] / max(context.trade_stats['total_trades'], 1) * 100
            if context.debug:
                print(f"{current_date} 年化收益率: {annualized_return:.1f}% (目标: {context.target_return}%)")
                print(f"  总收益: {total_return:.2%}, 胜率: {win_rate:.1f}%, 交易次数: {context.trade_stats['total_trades']}")
    
    # 动态风控
    consecutive_loss_days = sum(1 for p in context.performance_tracker[-5:] if p['daily_return'] < -0.01)
    risk_mode = (daily_return < -context.daily_loss_limit or consecutive_loss_days >= 3)
    
    if risk_mode and context.debug:
        print(f"{current_date} 触发风控保护")
    
    # 管理持仓
    manage_refined_positions(context, bar_dict, current_date, risk_mode)
    
    # 扫描机会
    if not risk_mode:
        scan_target_opportunities(context, bar_dict, current_date)

def update_market_regime(context):
    """更新市场状态"""
    try:
        benchmark = history_bars('000300.XSHG', 12, '1d', 'close', skip_suspended=True, include_now=True)
        if benchmark is None or len(benchmark) < 10:
            context.market_regime = 'normal'
            return
        
        # 多时间框架分析
        short_momentum = (benchmark[-1] / benchmark[-3] - 1)
        medium_momentum = (benchmark[-1] / benchmark[-7] - 1)
        volatility = np.std(benchmark[-8:]) / np.mean(benchmark[-8:])
        
        if short_momentum > 0.025 and medium_momentum > 0.04:
            context.market_regime = 'strong_bull'
        elif short_momentum < -0.025 and medium_momentum < -0.04:
            context.market_regime = 'strong_bear'
        elif volatility > 0.03:
            context.market_regime = 'volatile'
        elif abs(short_momentum) < 0.01 and abs(medium_momentum) < 0.02:
            context.market_regime = 'stable'
        else:
            context.market_regime = 'normal'
            
    except:
        context.market_regime = 'normal'

def manage_refined_positions(context, bar_dict, current_date, risk_mode=False):
    """精细化持仓管理"""
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
            # 风控模式
            if return_pct <= -context.stop_loss * 0.8:
                should_close, reason = True, f"风控止损，损失: {return_pct:.2%}"
        else:
            # 精确分层止盈
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
                should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            elif check_technical_exit_signal(stock, context):
                should_close, reason = True, f"技术退出，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason, return_pct))
        elif should_reduce:
            to_reduce.append((stock, reason))
    
    # 执行减仓 - 减至40%
    for stock, reason in to_reduce:
        position = context.portfolio.positions.get(stock)
        current_weight = position.market_value / context.portfolio.total_value
        new_weight = current_weight * 0.4
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
            context.trade_stats['consecutive_losses'] = 0
            context.trade_stats['max_single_return'] = max(context.trade_stats['max_single_return'], return_pct)
        else:
            context.trade_stats['consecutive_losses'] += 1
            context.trade_stats['max_single_loss'] = min(context.trade_stats['max_single_loss'], return_pct)
        
        if context.debug:
            win_rate = context.trade_stats['winning_trades'] / context.trade_stats['total_trades'] * 100
            avg_return = context.trade_stats['total_return'] / context.trade_stats['total_trades'] * 100
            print(f"{current_date} 平仓 {stock}: {reason}")
            if context.trade_stats['total_trades'] % 5 == 0:
                print(f"  统计: 胜率{win_rate:.1f}%, 平均{avg_return:.2f}%, 最大盈利{context.trade_stats['max_single_return']:.2%}")

def check_technical_exit_signal(stock, context):
    """检查技术退出信号"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 6, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 6:
            return False
        
        # 动量衰竭信号
        recent_momentum = (closes[-1] / closes[-3] - 1)
        if recent_momentum < -0.04:  # 连续3天跌4%
            return True
        
        # 成交量萎缩信号
        vol_ratio = volumes[-1] / np.mean(volumes[-5:])
        if vol_ratio < 0.5 and recent_momentum < 0:
            return True
        
        # 技术破位
        if len(closes) >= 6:
            ma5 = np.mean(closes[-5:])
            if closes[-1] < ma5 * 0.97:  # 跌破5日均线3%
                return True
        
        return False
    except:
        return False

def scan_target_opportunities(context, bar_dict, current_date):
    """扫描目标机会"""
    if context.debug:
        print(f"{current_date} 扫描目标机会 - 市场: {context.market_regime}")
    
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    if current_positions >= context.max_positions:
        return
    
    candidates = []
    scan_limit = 250
    scan_stocks = list(context.stock_universe)[:scan_limit]
    
    for stock in scan_stocks:
        if stock in context.positions:
            continue
        
        try:
            if not refined_filter(stock, bar_dict, context):
                continue
            
            # 三维评分系统
            momentum_score = calculate_refined_momentum(stock, context)
            volume_score = calculate_refined_volume(stock, context)
            quality_score = calculate_refined_quality(stock, context)
            
            # 检查各维度达标
            if (momentum_score >= context.min_momentum_score and 
                volume_score >= context.min_volume_score and
                quality_score >= context.min_quality_score):
                
                # 根据市场状态调整权重
                if context.market_regime == 'strong_bull':
                    total_score = momentum_score * 0.5 + volume_score * 0.3 + quality_score * 0.2
                elif context.market_regime == 'strong_bear':
                    total_score = quality_score * 0.5 + volume_score * 0.3 + momentum_score * 0.2
                else:
                    total_score = (momentum_score + volume_score + quality_score) / 3
                
                if total_score >= context.min_total_score:
                    strategy_type = determine_refined_strategy(stock, context)
                    candidates.append((stock, total_score, strategy_type))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现合格目标")
        return
    
    # 按评分排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    available_slots = context.max_positions - current_positions
    for i, (stock, score, strategy_type) in enumerate(candidates[:available_slots]):
        position_size = calculate_target_position_size(context, score, i)
        
        if position_size > 0.18:  # 最小18%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'strategy_type': strategy_type,
                'score': score,
                'profit_level': 0
            }
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 策略: {strategy_type}, 仓位: {position_size:.1%}, 评分: {score:.2f}")

def refined_filter(stock, bar_dict, context):
    """精细化筛选器"""
    try:
        price = bar_dict[stock].close
        
        if not (context.min_price <= price <= context.max_price):
            return False
        
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-10:] * volumes[-10:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 价格变动检查
        daily_change = abs(closes[-1] / closes[-2] - 1)
        if daily_change < 0.015:  # 至少1.5%变动
            return False
        
        # 避免极端下跌
        week_decline = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        if week_decline < -0.12:
            return False
        
        # 波动率检查
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
        if volatility > 0.15 or volatility < 0.02:
            return False
        
        return True
    except:
        return False

def calculate_refined_momentum(stock, context):
    """计算精细化动量评分"""
    try:
        closes = history_bars(stock, 15, '1d', 'close', skip_suspended=True, include_now=True)
        if closes is None or len(closes) < 10:
            return 0
        
        score = 0
        
        # 多时间框架动量 (70%)
        momentum_1d = (closes[-1] / closes[-2] - 1)
        momentum_3d = (closes[-1] / closes[-4] - 1)
        momentum_5d = (closes[-1] / closes[-6] - 1)
        
        if momentum_1d > context.momentum_threshold:
            score += 0.30
        elif momentum_1d > 0.04:
            score += 0.25
        elif momentum_1d > 0.025:
            score += 0.20
        elif momentum_1d > 0.015:
            score += 0.15
        
        if momentum_3d > context.momentum_threshold * 1.2:
            score += 0.25
        elif momentum_3d > 0.05:
            score += 0.20
        elif momentum_3d > 0.03:
            score += 0.15
        
        if momentum_5d > context.momentum_threshold * 1.5:
            score += 0.15
        elif momentum_5d > 0.06:
            score += 0.10
        elif momentum_5d > 0.03:
            score += 0.05
        
        # 加速度 (20%)
        if len(closes) >= 8:
            accel = momentum_3d - (closes[-4] / closes[-7] - 1)
            if accel > context.price_acceleration:
                score += 0.20
            elif accel > 0.025:
                score += 0.15
            elif accel > 0.01:
                score += 0.10
        
        # 相对强度 (10%)
        recent_high = np.max(closes[-8:])
        if closes[-1] >= recent_high:
            score += 0.10
        elif closes[-1] >= recent_high * 0.98:
            score += 0.08
        elif closes[-1] >= recent_high * 0.95:
            score += 0.05
        
        return min(score, 1.0)
    except:
        return 0

def calculate_refined_volume(stock, context):
    """计算精细化成交量评分"""
    try:
        closes = history_bars(stock, 12, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 12, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(volumes) < 8:
            return 0
        
        score = 0
        
        # 成交量突破 (60%)
        current_vol = volumes[-1]
        avg_vol_5d = np.mean(volumes[-6:-1])
        avg_vol_10d = np.mean(volumes[-11:-1])
        
        surge_5d = current_vol / avg_vol_5d if avg_vol_5d > 0 else 1
        surge_10d = current_vol / avg_vol_10d if avg_vol_10d > 0 else 1
        
        if surge_5d > context.volume_surge_min * 1.5:
            score += 0.35
        elif surge_5d > context.volume_surge_min:
            score += 0.30
        elif surge_5d > 2.0:
            score += 0.25
        elif surge_5d > 1.5:
            score += 0.15
        
        if surge_10d > context.volume_surge_min:
            score += 0.25
        elif surge_10d > 2.0:
            score += 0.20
        elif surge_10d > 1.5:
            score += 0.15
        
        # 价量配合 (40%)
        price_change = (closes[-1] / closes[-2] - 1)
        volume_change = (volumes[-1] / volumes[-2] - 1) if volumes[-2] > 0 else 0
        
        if price_change > 0.04 and volume_change > 1.5:
            score += 0.40
        elif price_change > 0.025 and volume_change > 1.0:
            score += 0.30
        elif price_change > 0.015 and volume_change > 0.5:
            score += 0.20
        elif price_change > 0 and volume_change > 0:
            score += 0.10
        
        return min(score, 1.0)
    except:
        return 0

def calculate_refined_quality(stock, context):
    """计算精细化质量评分"""
    try:
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        if closes is None or len(closes) < 15:
            return 0
        
        score = 0
        
        # 趋势质量 (50%)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma15 = np.mean(closes[-15:])
        
        if closes[-1] > ma5 > ma10 > ma15:
            score += 0.50
        elif closes[-1] > ma5 > ma10:
            score += 0.35
        elif closes[-1] > ma5:
            score += 0.25
        elif closes[-1] > ma10:
            score += 0.15
        
        # 波动率稳定性 (30%)
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
        if 0.03 <= volatility <= 0.08:
            score += 0.30
        elif 0.02 <= volatility <= 0.12:
            score += 0.20
        elif volatility <= 0.15:
            score += 0.10
        
        # 价格位置 (20%)
        recent_high = np.max(closes[-12:])
        recent_low = np.min(closes[-12:])
        price_position = (closes[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        if price_position > 0.8:
            score += 0.20
        elif price_position > 0.6:
            score += 0.15
        elif price_position > 0.4:
            score += 0.10
        elif price_position > 0.2:
            score += 0.05
        
        return min(score, 1.0)
    except:
        return 0

def determine_refined_strategy(stock, context):
    """确定精细化策略类型"""
    try:
        closes = history_bars(stock, 10, '1d', 'close', skip_suspended=True, include_now=True)
        if closes is None or len(closes) < 8:
            return 'momentum'
        
        momentum = (closes[-1] / closes[-4] - 1)
        volatility = np.std(closes[-6:]) / np.mean(closes[-6:])
        
        if momentum > 0.08 and volatility > 0.06:
            return 'explosive_momentum'
        elif momentum > 0.04:
            return 'steady_momentum'
        elif momentum < -0.02:
            return 'contrarian'
        else:
            return 'momentum'
    except:
        return 'momentum'

def calculate_target_position_size(context, score, position_index):
    """计算目标仓位大小"""
    base_size = context.base_position / context.max_positions
    
    # 评分权重
    score_multiplier = 1.0 + (score - 0.85) * 1.5
    
    # 市场状态权重
    if context.market_regime == 'strong_bull':
        regime_multiplier = 1.15
    elif context.market_regime == 'volatile':
        regime_multiplier = 1.10
    elif context.market_regime == 'strong_bear':
        regime_multiplier = 0.90
    else:
        regime_multiplier = 1.00
    
    # 优先级权重
    priority_multiplier = 1.4 if position_index == 0 else (1.2 if position_index == 1 else 1.0)
    
    position_size = base_size * score_multiplier * regime_multiplier * priority_multiplier
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0.18)