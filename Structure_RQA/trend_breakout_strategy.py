# -*- coding: utf-8 -*-
"""
趋势突破策略 v1.0
================
目标：年化收益率 25%+

全新策略架构：
1. 多时间框架分析 - 周线趋势 + 日线入场
2. 突破系统 - 捕捉强势突破行情
3. 动态仓位 - 根据信号强度分配仓位
4. 快速止盈 - 利润最大化
5. 严格止损 - 控制单笔损失
"""

from rqalpha.api import (
    all_instruments,
    history_bars,
    order_target_percent,
)
import numpy as np

__config__ = {
    "base": {
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "frequency": "1d",
        "accounts": {"stock": 1_000_000},
        "benchmark": "000300.XSHG",
    },
    "extra": {"log_level": "info"},
}

def init(context):
    """策略初始化"""
    # 核心参数
    context.max_positions = 4          # 集中持仓
    context.base_position = 0.90       # 高仓位运作
    context.single_position_limit = 0.30  # 单股最大仓位
    
    # 突破参数
    context.breakout_period = 20       # 突破周期
    context.volume_multiplier = 1.5    # 成交量放大倍数
    context.min_breakout_gain = 0.03   # 最小突破幅度
    
    # 风控参数
    context.stop_loss = 0.06           # 快速止损
    context.take_profit_1 = 0.12       # 第一目标
    context.take_profit_2 = 0.25       # 第二目标
    context.max_hold_days = 15         # 最大持有期
    
    # 筛选条件
    context.min_price = 8.0
    context.max_price = 100.0
    context.min_market_cap = 50e8      # 最小市值50亿
    context.min_volume_value = 100e6   # 日成交额1亿+
    
    # 调仓频率
    context.scan_frequency = 3         # 每3天扫描一次
    context.last_scan = None
    
    # 状态记录
    context.positions = {}
    context.candidate_pool = set()
    context.blacklist = set()
    context.performance_tracker = []
    context.weekly_trend_cache = {}
    context.debug = True
    
    print("趋势突破策略初始化完成 - 目标年化25%+")

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 记录表现
    nav = context.portfolio.total_value
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    })
    
    # 管理现有持仓
    manage_existing_positions(context, bar_dict, current_date)
    
    # 定期扫描新机会
    if should_scan_opportunities(context, current_date):
        scan_breakout_opportunities(context, bar_dict, current_date)

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
        profit_level = pos_info.get('profit_level', 0)
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        # 分级止盈策略
        should_close = False
        reason = ""
        new_weight = None
        
        if profit_level == 0 and return_pct >= context.take_profit_1:
            # 第一次达到止盈目标，减半仓位
            new_weight = position.market_value / context.portfolio.total_value * 0.5
            pos_info['profit_level'] = 1
            reason = f"第一层止盈，减仓50%，当前收益: {return_pct:.2%}"
            
        elif profit_level == 1 and return_pct >= context.take_profit_2:
            # 第二次达到止盈目标，全部卖出
            should_close, reason = True, f"第二层止盈，收益: {return_pct:.2%}"
            
        elif return_pct <= -context.stop_loss:
            # 止损
            should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            context.blacklist.add(stock)  # 加入黑名单
            
        elif hold_days >= context.max_hold_days:
            # 时间止损
            should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            
        elif check_trend_reversal(stock, context):
            # 趋势反转
            should_close, reason = True, f"趋势反转，收益: {return_pct:.2%}"
            
        if should_close:
            to_close.append((stock, reason))
        elif new_weight is not None:
            order_target_percent(stock, new_weight)
            if context.debug:
                print(f"{current_date} {reason}")
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_trend_reversal(stock, context):
    """检查趋势反转"""
    try:
        closes = history_bars(stock, 10, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 10, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 10:
            return False
        
        # 价格连续下跌
        if np.all(np.diff(closes[-4:]) < 0):
            return True
        
        # 成交量萎缩且价格下跌
        vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-10:])
        price_change = (closes[-1] / closes[-4] - 1)
        
        if vol_ratio < 0.7 and price_change < -0.02:
            return True
        
        return False
    except:
        return False

def should_scan_opportunities(context, current_date):
    """判断是否需要扫描机会"""
    # 如果仓位未满，更频繁扫描
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    if current_positions < context.max_positions:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= 1  # 仓位未满时每天扫描
    else:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= context.scan_frequency

def scan_breakout_opportunities(context, bar_dict, current_date):
    """扫描突破机会"""
    if context.debug:
        print(f"{current_date} 开始扫描突破机会")
    
    # 获取市场整体状态
    market_strength = analyze_market_strength(context)
    
    # 如果市场疲弱，降低开仓积极性
    if market_strength < 0.3:
        if context.debug:
            print(f"{current_date} 市场疲弱({market_strength:.2f})，暂停开新仓")
        context.last_scan = current_date
        return
    
    # 扫描候选股票
    candidates = find_breakout_candidates(context, bar_dict, market_strength)
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现突破机会")
        context.last_scan = current_date
        return
    
    # 按信号强度排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    opened_positions = 0
    for stock, signal_strength, breakout_info in candidates[:available_slots]:
        if opened_positions >= available_slots:
            break
        
        # 计算仓位大小
        position_size = calculate_position_size(context, signal_strength, market_strength)
        
        if position_size > 0.05:  # 最小5%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'signal_strength': signal_strength,
                'breakout_info': breakout_info,
                'profit_level': 0
            }
            
            opened_positions += 1
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 仓位: {position_size:.1%}, 信号强度: {signal_strength:.2f}")
    
    context.last_scan = current_date

def analyze_market_strength(context):
    """分析市场整体强度"""
    try:
        closes = history_bars("000300.XSHG", 40, '1d', 'close', include_now=True)
        volumes = history_bars("000300.XSHG", 20, '1d', 'volume', include_now=True)
        
        if closes is None or len(closes) < 30:
            return 0.5
        
        # 多时间框架趋势分析
        current = closes[-1]
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma40 = np.mean(closes[-40:]) if len(closes) >= 40 else ma20
        
        # 趋势强度
        trend_score = 0
        if current > ma5 > ma10 > ma20 > ma40:
            trend_score = 1.0
        elif current > ma5 > ma10 > ma20:
            trend_score = 0.8
        elif current > ma5 > ma10:
            trend_score = 0.6
        elif current > ma5:
            trend_score = 0.4
        else:
            trend_score = 0.2
        
        # 动量分析
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        momentum_20d = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
        
        momentum_score = 0.5
        if momentum_5d > 0.02 and momentum_20d > 0.05:
            momentum_score = 1.0
        elif momentum_5d > 0 and momentum_20d > 0:
            momentum_score = 0.7
        elif momentum_5d > 0:
            momentum_score = 0.6
        elif momentum_5d > -0.02:
            momentum_score = 0.4
        else:
            momentum_score = 0.2
        
        # 成交量确认
        volume_score = 0.5
        if volumes is not None and len(volumes) >= 10:
            vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-10:])
            if vol_ratio > 1.3:
                volume_score = 1.0
            elif vol_ratio > 1.1:
                volume_score = 0.7
            elif vol_ratio > 0.9:
                volume_score = 0.5
            else:
                volume_score = 0.3
        
        # 综合评分
        market_strength = (trend_score * 0.5 + momentum_score * 0.3 + volume_score * 0.2)
        
        return market_strength
    
    except:
        return 0.5

def find_breakout_candidates(context, bar_dict, market_strength):
    """寻找突破候选股票"""
    instruments = all_instruments('CS')
    candidates = []
    
    count = 0
    for _, stock_info in instruments.iterrows():
        count += 1
        if count > 500:  # 扫描前500只股票
            break
        
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 跳过特殊股票和黑名单
        if any(x in symbol for x in ['ST', '*ST', '退']) or stock in context.blacklist:
            continue
        
        # 跳过已持有股票
        if stock in context.positions:
            continue
        
        try:
            # 基础筛选
            if not basic_breakout_filter(stock, bar_dict, context):
                continue
            
            # 突破信号检测
            signal_strength, breakout_info = detect_breakout_signal(stock, context, market_strength)
            
            if signal_strength > 0.6:  # 信号强度阈值
                candidates.append((stock, signal_strength, breakout_info))
        
        except:
            continue
    
    return candidates

def basic_breakout_filter(stock, bar_dict, context):
    """基础突破筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取数据
        closes = history_bars(stock, 60, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 30, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 30:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-10:] * volumes[-10:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 趋势基础
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else ma20
        
        # 要求在上升趋势中
        if ma20 <= ma60 * 1.02:  # 20日线要明显高于60日线
            return False
        
        # 避免连续大涨股票
        recent_gain = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        if recent_gain > 0.20:  # 避免短期涨幅过大
            return False
        
        return True
    
    except:
        return False

def detect_breakout_signal(stock, context, market_strength):
    """检测突破信号"""
    try:
        closes = history_bars(stock, context.breakout_period + 10, '1d', 'close', 
                            skip_suspended=True, include_now=True)
        volumes = history_bars(stock, context.breakout_period + 10, '1d', 'volume', 
                             skip_suspended=True, include_now=True)
        highs = history_bars(stock, context.breakout_period + 10, '1d', 'high', 
                           skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or highs is None:
            return 0, {}
        
        if len(closes) < context.breakout_period + 5:
            return 0, {}
        
        current_price = closes[-1]
        current_volume = volumes[-1]
        
        # 计算突破基准
        breakout_base = np.max(highs[-context.breakout_period-1:-1])  # 排除今日
        volume_base = np.mean(volumes[-20:-1])  # 排除今日
        
        # 突破条件检查
        breakout_gain = (current_price / breakout_base - 1)
        volume_ratio = current_volume / volume_base
        
        signal_strength = 0
        breakout_info = {
            'breakout_gain': breakout_gain,
            'volume_ratio': volume_ratio,
            'breakout_base': breakout_base
        }
        
        # 价格突破
        if breakout_gain >= context.min_breakout_gain:
            signal_strength += 0.4
            
            # 突破幅度奖励
            if breakout_gain > 0.05:
                signal_strength += 0.1
            if breakout_gain > 0.08:
                signal_strength += 0.1
        else:
            return 0, breakout_info
        
        # 成交量确认
        if volume_ratio >= context.volume_multiplier:
            signal_strength += 0.3
            
            # 成交量放大奖励
            if volume_ratio > 2.0:
                signal_strength += 0.1
            if volume_ratio > 3.0:
                signal_strength += 0.1
        else:
            signal_strength += max(0, (volume_ratio - 1.0) * 0.3)
        
        # 技术形态加分
        if len(closes) >= 10:
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            
            if current_price > ma5 > ma10:
                signal_strength += 0.1
            
            # 整理突破
            consolidation_range = (np.max(closes[-10:-1]) / np.min(closes[-10:-1]) - 1)
            if 0.05 <= consolidation_range <= 0.15:  # 合理整理幅度
                signal_strength += 0.1
        
        # 市场环境调整
        signal_strength *= (0.5 + 0.5 * market_strength)
        
        return min(signal_strength, 1.0), breakout_info
    
    except:
        return 0, {}

def calculate_position_size(context, signal_strength, market_strength):
    """计算仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 根据信号强度调整
    signal_multiplier = 0.5 + signal_strength
    
    # 根据市场强度调整
    market_multiplier = 0.7 + 0.6 * market_strength
    
    # 计算最终仓位
    position_size = base_size * signal_multiplier * market_multiplier
    
    # 限制单股最大仓位
    position_size = min(position_size, context.single_position_limit)
    
    # 确保总仓位不超限
    current_total_position = sum(
        pos.market_value for pos in context.portfolio.positions.values() 
        if pos.quantity > 0
    ) / context.portfolio.total_value
    
    available_position = context.base_position - current_total_position
    position_size = min(position_size, available_position)
    
    return max(position_size, 0)