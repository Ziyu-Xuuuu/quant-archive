# -*- coding: utf-8 -*-
"""
高频波段策略 2025 v1.0
====================
针对2025年市场特点的高频波段交易策略

核心逻辑：
1. 快速波段 - 2-8天持有周期，捕捉短期波动
2. 技术驱动 - 基于技术指标和量价关系
3. 快进快出 - 严格止盈止损，控制回撤
4. 高换手率 - 适应2025年市场活跃度
"""

from rqalpha.api import (
    all_instruments,
    history_bars,
    order_target_percent,
)
import numpy as np

__config__ = {
    "base": {
        "start_date": "2025-01-01",
        "end_date": "2025-07-28",
        "frequency": "1d",
        "accounts": {"stock": 1_000_000},
        "benchmark": "000300.XSHG",
    },
    "extra": {"log_level": "info"},
}

def init(context):
    """策略初始化"""
    # 核心参数
    context.max_positions = 8          # 最多8只股票
    context.base_position = 0.88       # 高仓位运作
    context.single_position_limit = 0.15  # 单股最大15%
    
    # 波段参数
    context.min_hold_days = 2          # 最少持有2天
    context.max_hold_days = 8          # 最多持有8天
    context.rsi_oversold = 25          # RSI超卖线
    context.rsi_overbought = 75        # RSI超买线
    context.volume_surge_ratio = 2.0   # 成交量激增比例
    
    # 风控参数
    context.stop_loss = 0.05           # 快速5%止损
    context.take_profit = 0.12         # 12%止盈
    context.trailing_stop = 0.03       # 3%追踪止损
    
    # 筛选条件
    context.min_price = 8.0
    context.max_price = 80.0
    context.min_market_cap = 60e8      # 最小市值60亿
    context.min_volume_value = 80e6    # 日成交额8000万+
    context.max_volatility = 0.12      # 最大波动率12%
    
    # 技术指标参数
    context.rsi_period = 14
    context.ma_short = 5
    context.ma_long = 20
    context.bb_period = 20
    context.bb_std = 2
    
    # 状态记录
    context.positions = {}
    context.candidate_pool = []
    context.scan_frequency = 1         # 每天扫描
    context.last_scan = None
    context.performance_tracker = []
    context.debug = True
    
    print("高频波段策略初始化完成 - 目标快进快出")

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 记录表现
    nav = context.portfolio.total_value
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'positions': current_positions,
        'turnover': calculate_daily_turnover(context)
    })
    
    # 管理现有持仓
    manage_swing_positions(context, bar_dict, current_date)
    
    # 每天扫描新机会
    if should_scan_swing_opportunities(context, current_date):
        scan_swing_opportunities(context, bar_dict, current_date)

def calculate_daily_turnover(context):
    """计算日换手率"""
    try:
        if len(context.performance_tracker) < 2:
            return 0
        
        yesterday_positions = set()
        today_positions = set()
        
        # 获取昨日持仓
        for stock in context.positions.keys():
            position = context.portfolio.positions.get(stock)
            if position and position.quantity > 0:
                today_positions.add(stock)
        
        # 简化计算：假设今日新增持仓为换手
        if hasattr(context, 'yesterday_positions'):
            changed_positions = len(today_positions.symmetric_difference(context.yesterday_positions))
            turnover = changed_positions / max(len(today_positions), 1)
        else:
            turnover = 0
        
        context.yesterday_positions = today_positions.copy()
        return turnover
    except:
        return 0

def manage_swing_positions(context, bar_dict, current_date):
    """管理波段持仓"""
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
        highest_price = pos_info.get('highest_price', entry_price)
        
        # 更新最高价
        if current_price > highest_price:
            pos_info['highest_price'] = current_price
            highest_price = current_price
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        trailing_loss = (highest_price - current_price) / highest_price
        
        # 退出条件
        should_close = False
        reason = ""
        
        if hold_days >= context.min_hold_days:  # 最少持有期后才能卖出
            if return_pct >= context.take_profit:
                should_close, reason = True, f"止盈，收益: {return_pct:.2%}"
            elif trailing_loss >= context.trailing_stop and return_pct > 0.02:
                should_close, reason = True, f"追踪止损，收益: {return_pct:.2%}"
            elif return_pct <= -context.stop_loss:
                should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            elif hold_days >= context.max_hold_days:
                should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            elif check_technical_exit_signal(stock, context):
                should_close, reason = True, f"技术信号退出，收益: {return_pct:.2%}"
        elif return_pct <= -context.stop_loss:
            # 即使在最少持有期内，也要执行硬止损
            should_close, reason = True, f"硬止损，损失: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason))
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_technical_exit_signal(stock, context):
    """检查技术退出信号"""
    try:
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return False
        
        current = closes[-1]
        
        # RSI超买
        rsi = calculate_rsi(closes, context.rsi_period)
        if rsi > context.rsi_overbought:
            return True
        
        # 布林带上轨突破后回落
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, context.bb_period, context.bb_std)
        if len(closes) >= 3:
            if closes[-2] > bb_upper and current < bb_upper:
                return True
        
        # 成交量萎缩
        if len(volumes) >= 8:
            vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-8:])
            if vol_ratio < 0.6:
                return True
        
        return False
    except:
        return False

def should_scan_swing_opportunities(context, current_date):
    """判断是否需要扫描波段机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    # 每天都扫描，但仓位满时降低开仓积极性
    if context.last_scan is None:
        return True
    
    days_since_scan = (current_date - context.last_scan).days
    return days_since_scan >= context.scan_frequency

def scan_swing_opportunities(context, bar_dict, current_date):
    """扫描波段交易机会"""
    if context.debug and len(context.positions) < context.max_positions:
        print(f"{current_date} 扫描波段机会")
    
    # 获取市场情绪
    market_sentiment = analyze_market_sentiment(context)
    
    # 扫描候选股票
    candidates = find_swing_candidates(context, bar_dict, market_sentiment, current_date)
    
    if not candidates:
        context.last_scan = current_date
        return
    
    # 按信号强度排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    opened_positions = 0
    for stock, signal_strength, signal_info in candidates[:available_slots]:
        if opened_positions >= available_slots:
            break
        
        # 计算仓位大小
        position_size = calculate_swing_position_size(context, signal_strength, market_sentiment)
        
        if position_size > 0.05:  # 最小5%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'highest_price': bar_dict[stock].close,
                'signal_strength': signal_strength,
                'signal_type': signal_info.get('type', 'unknown')
            }
            
            opened_positions += 1
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 仓位: {position_size:.1%}, "
                      f"信号: {signal_info.get('type', 'unknown')}, 强度: {signal_strength:.2f}")
    
    context.last_scan = current_date

def analyze_market_sentiment(context):
    """分析市场情绪"""
    try:
        closes = history_bars("000300.XSHG", 20, '1d', 'close', include_now=True)
        volumes = history_bars("000300.XSHG", 15, '1d', 'volume', include_now=True)
        
        if closes is None or len(closes) < 15:
            return 0.5
        
        # 价格动量
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        momentum_score = 0.5 + momentum_5d * 5  # 转换为0-1分数
        momentum_score = max(0, min(1, momentum_score))
        
        # 成交量活跃度
        volume_score = 0.5
        if volumes is not None and len(volumes) >= 10:
            vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-10:])
            volume_score = min(1, vol_ratio / 2)
        
        # 波动率
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
        volatility_score = min(1, volatility * 10)  # 波动率越高，交易机会越多
        
        # 综合情绪
        sentiment = (momentum_score * 0.4 + volume_score * 0.3 + volatility_score * 0.3)
        return max(0.2, min(0.8, sentiment))
    
    except:
        return 0.5

def find_swing_candidates(context, bar_dict, market_sentiment, current_date):
    """寻找波段交易候选股票"""
    instruments = all_instruments('CS')
    candidates = []
    
    count = 0
    for _, stock_info in instruments.iterrows():
        count += 1
        if count > 400:  # 扫描前400只股票
            break
        
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        
        # 跳过特殊股票
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
        
        # 跳过已持有股票
        if stock in context.positions:
            continue
        
        try:
            # 基础筛选
            if not basic_swing_filter(stock, bar_dict, context):
                continue
            
            # 技术信号检测
            signal_strength, signal_info = detect_swing_signals(stock, context, market_sentiment)
            
            if signal_strength > 0.7:  # 信号强度阈值
                candidates.append((stock, signal_strength, signal_info))
        
        except:
            continue
    
    return candidates

def basic_swing_filter(stock, bar_dict, context):
    """基础波段筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取数据
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 25:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-10:] * volumes[-10:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 波动率检查
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
        if volatility > context.max_volatility:
            return False
        
        # 避免单边下跌
        decline_days = 0
        for i in range(1, min(6, len(closes))):
            if closes[-i] < closes[-i-1]:
                decline_days += 1
        
        if decline_days >= 4:  # 连续4天以上下跌
            return False
        
        return True
    
    except:
        return False

def detect_swing_signals(stock, context, market_sentiment):
    """检测波段交易信号"""
    try:
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 25, '1d', 'volume', skip_suspended=True, include_now=True)
        highs = history_bars(stock, 30, '1d', 'high', skip_suspended=True, include_now=True)
        lows = history_bars(stock, 30, '1d', 'low', skip_suspended=True, include_now=True)
        
        if not all([closes is not None, volumes is not None, 
                   highs is not None, lows is not None]):
            return 0, {}
        
        if len(closes) < 25:
            return 0, {}
        
        current = closes[-1]
        signal_strength = 0
        signal_info = {'type': 'none'}
        
        # 1. RSI超卖反弹信号
        rsi = calculate_rsi(closes, context.rsi_period)
        if rsi < context.rsi_oversold:
            # 检查是否有反弹迹象
            if closes[-1] > closes[-2]:  # 价格开始反弹
                signal_strength += 0.4
                signal_info['type'] = 'rsi_oversold_bounce'
                signal_info['rsi'] = rsi
        
        # 2. 布林带下轨支撑信号
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, context.bb_period, context.bb_std)
        if current <= bb_lower * 1.02:  # 接近或触及下轨
            signal_strength += 0.3
            if signal_info['type'] == 'none':
                signal_info['type'] = 'bollinger_support'
        
        # 3. 成交量激增信号
        if len(volumes) >= 15:
            vol_recent = volumes[-1]
            vol_avg = np.mean(volumes[-10:-1])  # 排除当日
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
            
            if vol_ratio >= context.volume_surge_ratio:
                signal_strength += 0.35
                if signal_info['type'] == 'none':
                    signal_info['type'] = 'volume_surge'
                signal_info['volume_ratio'] = vol_ratio
        
        # 4. 均线支撑信号
        if len(closes) >= 20:
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            
            # 价格回调至均线附近
            if ma20 < current <= ma5 * 1.02 and ma5 > ma20:
                signal_strength += 0.25
                if signal_info['type'] == 'none':
                    signal_info['type'] = 'ma_support'
        
        # 5. 双底或V型反转形态
        if len(lows) >= 15:
            recent_low = np.min(lows[-5:])
            prev_low = np.min(lows[-15:-5])
            
            # 双底形态
            if abs(recent_low / prev_low - 1) < 0.03:  # 两个低点接近
                if current > recent_low * 1.02:  # 价格开始反弹
                    signal_strength += 0.2
                    if signal_info['type'] == 'none':
                        signal_info['type'] = 'double_bottom'
        
        # 市场环境调整
        signal_strength *= (0.7 + 0.6 * market_sentiment)
        
        return min(signal_strength, 1.0), signal_info
    
    except:
        return 0, {}

def calculate_rsi(closes, period=14):
    """计算RSI指标"""
    if len(closes) < period + 1:
        return 50
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(closes, period=20, std_dev=2):
    """计算布林带"""
    if len(closes) < period:
        middle = np.mean(closes)
        std = np.std(closes)
    else:
        middle = np.mean(closes[-period:])
        std = np.std(closes[-period:])
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower

def calculate_swing_position_size(context, signal_strength, market_sentiment):
    """计算波段仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 根据信号强度调整
    signal_multiplier = 0.6 + 0.8 * signal_strength
    
    # 根据市场情绪调整
    market_multiplier = 0.8 + 0.4 * market_sentiment
    
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