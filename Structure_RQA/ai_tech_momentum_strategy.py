# -*- coding: utf-8 -*-
"""
AI科技动量策略 2025 v1.0
========================
针对2025年AI科技热潮优化的策略

核心逻辑：
1. 专注AI/科技板块 - 捕捉2025年AI主题投资机会
2. 动量追踪 - 识别强势上涨股票
3. 快速轮动 - 适应科技股高波动特性
4. 智能止盈止损 - 保护利润，限制损失
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
    context.max_positions = 5          # 集中持仓5只
    context.base_position = 0.95       # 超高仓位运作
    context.single_position_limit = 0.25  # 单股最大25%
    
    # AI科技关键词
    context.ai_keywords = [
        '人工智能', '机器学习', '深度学习', '算法', '芯片', 
        '半导体', '云计算', '大数据', '物联网', '5G',
        '软件开发', '信息技术', '电子信息', '通信技术',
        '自动驾驶', '机器人', '智能制造', '数字经济',
        '区块链', '量子计算', '边缘计算', '神经网络'
    ]
    
    # 动量参数
    context.momentum_period = 5        # 动量计算周期
    context.min_momentum = 0.03        # 最小动量阈值
    context.volume_surge_ratio = 1.8   # 成交量激增比例
    
    # 风控参数
    context.stop_loss = 0.08           # 8%止损
    context.take_profit_1 = 0.15       # 第一层15%止盈
    context.take_profit_2 = 0.30       # 第二层30%止盈
    context.max_hold_days = 12         # 最大持有12天
    
    # 筛选条件
    context.min_price = 5.0
    context.max_price = 150.0
    context.min_market_cap = 30e8      # 最小市值30亿
    context.min_volume_value = 50e6    # 日成交额5000万+
    
    # 状态记录
    context.positions = {}
    context.ai_stock_pool = set()
    context.performance_tracker = []
    context.scan_frequency = 2         # 每2天扫描一次
    context.last_scan = None
    context.debug = True
    
    # 初始化AI股票池
    update_ai_stock_pool(context)
    print(f"AI科技动量策略初始化完成 - AI股票池: {len(context.ai_stock_pool)}只")

def update_ai_stock_pool(context):
    """更新AI股票池"""
    instruments = all_instruments('CS')
    ai_stocks = set()
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        company_name = stock_info.get('display_name', '')
        industry = stock_info.get('industry_name', '')
        
        # 跳过特殊股票
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
        
        # 检查是否为AI科技股
        text_to_check = f"{company_name} {industry}".lower()
        for keyword in context.ai_keywords:
            if keyword in text_to_check:
                ai_stocks.add(stock)
                break
    
    context.ai_stock_pool = ai_stocks

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
    manage_positions(context, bar_dict, current_date)
    
    # 定期扫描新机会
    if should_scan_new_opportunities(context, current_date):
        scan_ai_momentum_opportunities(context, bar_dict, current_date)

def manage_positions(context, bar_dict, current_date):
    """管理持仓"""
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
            # 第一层止盈，减半仓位
            current_weight = position.market_value / context.portfolio.total_value
            new_weight = current_weight * 0.5
            pos_info['profit_level'] = 1
            reason = f"第一层止盈减仓，收益: {return_pct:.2%}"
            
        elif profit_level == 1 and return_pct >= context.take_profit_2:
            # 第二层止盈，全部卖出
            should_close, reason = True, f"第二层止盈，收益: {return_pct:.2%}"
            
        elif return_pct <= -context.stop_loss:
            # 止损
            should_close, reason = True, f"止损，损失: {return_pct:.2%}"
            
        elif hold_days >= context.max_hold_days:
            # 时间止损
            should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
            
        elif check_momentum_weakening(stock, context):
            # 动量衰竭
            should_close, reason = True, f"动量衰竭，收益: {return_pct:.2%}"
        
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

def check_momentum_weakening(stock, context):
    """检查动量是否衰竭"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 8, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 8:
            return False
        
        # 连续3天下跌
        recent_changes = np.diff(closes[-4:])
        if np.sum(recent_changes < 0) >= 3:
            return True
        
        # 成交量萎缩超过50%
        vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-8:])
        if vol_ratio < 0.5:
            return True
        
        return False
    except:
        return False

def should_scan_new_opportunities(context, current_date):
    """判断是否需要扫描新机会"""
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    
    if current_positions < context.max_positions:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= 1  # 仓位未满每天扫描
    else:
        if context.last_scan is None:
            return True
        days_since_scan = (current_date - context.last_scan).days
        return days_since_scan >= context.scan_frequency

def scan_ai_momentum_opportunities(context, bar_dict, current_date):
    """扫描AI动量机会"""
    if context.debug:
        print(f"{current_date} 扫描AI动量机会")
    
    candidates = []
    
    # 在AI股票池中寻找机会
    for stock in list(context.ai_stock_pool)[:200]:  # 限制扫描数量
        if stock in context.positions:
            continue
        
        try:
            # 基础筛选
            if not basic_ai_filter(stock, bar_dict, context):
                continue
            
            # 动量信号检测
            momentum_score = calculate_momentum_score(stock, context)
            
            if momentum_score > 0.7:  # 动量阈值
                candidates.append((stock, momentum_score))
        
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 未发现AI动量机会")
        context.last_scan = current_date
        return
    
    # 按动量评分排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 开仓
    current_positions = len([p for p in context.portfolio.positions.values() if p.quantity > 0])
    available_slots = context.max_positions - current_positions
    
    for i, (stock, score) in enumerate(candidates[:available_slots]):
        # 计算仓位大小
        position_size = calculate_ai_position_size(context, score, i)
        
        if position_size > 0.05:  # 最小5%仓位
            order_target_percent(stock, position_size)
            
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'momentum_score': score,
                'profit_level': 0
            }
            
            if context.debug:
                print(f"{current_date} 开仓 {stock}, 仓位: {position_size:.1%}, 动量评分: {score:.2f}")
    
    context.last_scan = current_date

def basic_ai_filter(stock, bar_dict, context):
    """基础AI股票筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取数据
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-10:] * volumes[-10:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 避免连续大跌
        recent_decline = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        if recent_decline < -0.20:
            return False
        
        return True
    
    except:
        return False

def calculate_momentum_score(stock, context):
    """计算动量评分"""
    try:
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return 0
        
        current = closes[-1]
        score = 0
        
        # 1. 价格动量 (50%)
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        
        if momentum_5d > context.min_momentum:
            score += 0.25
            if momentum_5d > 0.06:
                score += 0.15
            if momentum_5d > 0.10:
                score += 0.10
        
        if momentum_3d > 0.02:
            score += 0.15
        
        if momentum_1d > 0.01:
            score += 0.10
        
        # 2. 成交量动量 (30%)
        if len(volumes) >= 10:
            vol_recent = np.mean(volumes[-3:])
            vol_base = np.mean(volumes[-10:])
            vol_ratio = vol_recent / vol_base if vol_base > 0 else 1
            
            if vol_ratio > context.volume_surge_ratio:
                score += 0.30
            elif vol_ratio > 1.3:
                score += 0.20
            elif vol_ratio > 1.0:
                score += 0.10
        
        # 3. 技术形态 (20%)
        if len(closes) >= 15:
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma15 = np.mean(closes[-15:])
            
            # 均线多头排列
            if current > ma5 > ma10 > ma15:
                score += 0.20
            elif current > ma5 > ma10:
                score += 0.15
            elif current > ma5:
                score += 0.10
        
        return min(score, 1.0)
    
    except:
        return 0

def calculate_ai_position_size(context, momentum_score, position_index):
    """计算AI股票仓位大小"""
    # 基础仓位
    base_size = context.base_position / context.max_positions
    
    # 根据动量评分调整 (评分越高仓位越大)
    momentum_multiplier = 0.6 + 0.8 * momentum_score
    
    # 首选股票获得更大仓位
    priority_multiplier = 1.2 if position_index == 0 else (1.1 if position_index == 1 else 1.0)
    
    # 计算最终仓位
    position_size = base_size * momentum_multiplier * priority_multiplier
    
    # 限制单股最大仓位
    position_size = min(position_size, context.single_position_limit)
    
    return max(position_size, 0)