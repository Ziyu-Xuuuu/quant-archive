# -*- coding: utf-8 -*-
"""
行业轮动策略 v1.0
================
目标：年化收益率 25%+

全新策略架构：
1. 行业轮动 - 识别强势行业并轮动配置
2. 均值回归 - 短期超跌后的反弹机会
3. 强势股追踪 - 行业内最强个股
4. 快速切换 - 及时调整行业配置
5. 风险分散 - 多行业配置降低风险
"""

from rqalpha.api import (
    all_instruments,
    history_bars,
    order_target_percent,
    sector,
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
    context.max_sectors = 3            # 最多3个行业
    context.stocks_per_sector = 2      # 每行业2只股票
    context.total_stocks = 6           # 总共6只股票
    context.base_position = 0.85       # 高仓位运作
    
    # 轮动参数
    context.rotation_period = 5        # 轮动周期（天）
    context.sector_momentum_period = 10 # 行业动量计算周期
    context.rebalance_threshold = 0.03  # 重新平衡阈值
    
    # 个股选择参数
    context.mean_reversion_period = 5   # 均值回归周期
    context.oversold_threshold = -0.08  # 超跌阈值
    context.min_rebound_strength = 0.02 # 最小反弹强度
    
    # 风控参数
    context.stop_loss = 0.07           # 个股止损
    context.sector_stop_loss = 0.10    # 行业止损
    context.take_profit = 0.18         # 个股止盈
    context.max_hold_days = 20         # 最大持有天数
    
    # 筛选条件
    context.min_price = 10.0
    context.max_price = 80.0
    context.min_market_cap = 80e8      # 最小市值80亿
    context.min_volume_value = 80e6    # 日成交额8000万+
    
    # 行业定义
    context.sectors = {
        'technology': ['软件服务', '电子元件', '电子信息', '通信设备'],
        'healthcare': ['医疗保健', '生物制药', '医疗器械', '中药'],
        'consumer': ['食品饮料', '家用电器', '纺织服装', '商贸代理'],
        'finance': ['银行', '保险', '证券', '信托'],
        'materials': ['钢铁', '有色金属', '化工原料', '建筑材料'],
        'energy': ['石油', '天然气', '煤炭开采', '电力'],
        'industrials': ['机械制造', '运输设备', '建筑工程', '国防军工'],
        'real_estate': ['房地产', '建筑装饰', '园林工程']
    }
    
    # 状态记录
    context.positions = {}
    context.sector_performance = {}
    context.last_rotation = None
    context.sector_rankings = []
    context.performance_history = []
    context.debug = True
    
    print("行业轮动策略初始化完成 - 目标年化25%+")

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 记录表现
    nav = context.portfolio.total_value
    context.performance_history.append({
        'date': current_date,
        'nav': nav
    })
    
    # 管理现有持仓
    manage_positions(context, bar_dict, current_date)
    
    # 定期进行行业轮动
    if should_rotate_sectors(context, current_date):
        rotate_sectors(context, bar_dict, current_date)

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
        sector_name = pos_info['sector']
        
        return_pct = (current_price / entry_price - 1)
        hold_days = (current_date - entry_date).days
        
        # 退出条件
        should_close = False
        reason = ""
        
        if return_pct >= context.take_profit:
            should_close, reason = True, f"止盈，收益: {return_pct:.2%}"
        elif return_pct <= -context.stop_loss:
            should_close, reason = True, f"止损，损失: {return_pct:.2%}"
        elif hold_days >= context.max_hold_days:
            should_close, reason = True, f"时间止损，收益: {return_pct:.2%}"
        elif check_sector_weakness(sector_name, context):
            should_close, reason = True, f"行业转弱，收益: {return_pct:.2%}"
        elif check_individual_weakness(stock, context):
            should_close, reason = True, f"个股转弱，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason))
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_sector_weakness(sector_name, context):
    """检查行业是否转弱"""
    try:
        # 获取行业近期表现
        sector_stocks = get_sector_stocks(sector_name, context)
        if len(sector_stocks) < 5:
            return False
        
        # 计算行业平均表现
        sector_returns = []
        for stock in sector_stocks[:10]:  # 取前10只股票
            try:
                closes = history_bars(stock, 5, '1d', 'close', skip_suspended=True, include_now=True)
                if closes is not None and len(closes) >= 5:
                    ret = (closes[-1] / closes[-5] - 1)
                    sector_returns.append(ret)
            except:
                continue
        
        if len(sector_returns) >= 3:
            avg_return = np.mean(sector_returns)
            return avg_return < -0.05  # 行业平均跌幅超过5%
        
        return False
    except:
        return False

def check_individual_weakness(stock, context):
    """检查个股是否转弱"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 8, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 8:
            return False
        
        # 连续下跌
        recent_changes = np.diff(closes[-4:])
        if np.sum(recent_changes < 0) >= 3:
            return True
        
        # 成交量萎缩且价格下跌
        vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-8:])
        price_change = (closes[-1] / closes[-4] - 1)
        
        if vol_ratio < 0.6 and price_change < -0.03:
            return True
        
        return False
    except:
        return False

def should_rotate_sectors(context, current_date):
    """判断是否需要行业轮动"""
    if context.last_rotation is None:
        return True
    
    days_since_rotation = (current_date - context.last_rotation).days
    
    # 定期轮动
    if days_since_rotation >= context.rotation_period:
        return True
    
    # 应急轮动：当前持仓行业表现过差
    current_sectors = set()
    for stock, pos_info in context.positions.items():
        current_sectors.add(pos_info['sector'])
    
    for sector in current_sectors:
        if check_sector_weakness(sector, context):
            return True
    
    return False

def rotate_sectors(context, bar_dict, current_date):
    """执行行业轮动"""
    if context.debug:
        print(f"{current_date} 开始行业轮动")
    
    # 分析行业强度
    sector_rankings = analyze_sector_strength(context)
    context.sector_rankings = sector_rankings
    
    if not sector_rankings:
        if context.debug:
            print(f"{current_date} 无法获取行业数据")
        return
    
    # 选择最强的行业
    top_sectors = sector_rankings[:context.max_sectors]
    
    if context.debug:
        print(f"{current_date} 选择行业: {[s[0] for s in top_sectors]}")
    
    # 获取当前持仓行业
    current_sector_positions = {}
    for stock, pos_info in context.positions.items():
        sector = pos_info['sector']
        if sector not in current_sector_positions:
            current_sector_positions[sector] = []
        current_sector_positions[sector].append(stock)
    
    # 清理不在前列的行业
    sectors_to_keep = set(s[0] for s in top_sectors)
    for sector in list(current_sector_positions.keys()):
        if sector not in sectors_to_keep:
            for stock in current_sector_positions[sector]:
                order_target_percent(stock, 0)
                context.positions.pop(stock, None)
                if context.debug:
                    print(f"{current_date} 清理行业 {sector}: {stock}")
    
    # 为每个选中行业选择股票
    target_weight_per_sector = context.base_position / len(top_sectors)
    target_weight_per_stock = target_weight_per_sector / context.stocks_per_sector
    
    for sector_name, sector_score in top_sectors:
        select_sector_stocks(context, bar_dict, sector_name, target_weight_per_stock, current_date)
    
    context.last_rotation = current_date

def analyze_sector_strength(context):
    """分析行业强度"""
    sector_scores = []
    
    for sector_name in context.sectors.keys():
        try:
            sector_stocks = get_sector_stocks(sector_name, context)
            if len(sector_stocks) < 3:
                continue
            
            # 计算行业平均表现
            returns_5d = []
            returns_10d = []
            volume_ratios = []
            
            for stock in sector_stocks[:20]:  # 取前20只股票
                try:
                    closes = history_bars(stock, 15, '1d', 'close', skip_suspended=True, include_now=True)
                    volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
                    
                    if closes is not None and len(closes) >= 15:
                        ret_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                        ret_10d = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
                        returns_5d.append(ret_5d)
                        returns_10d.append(ret_10d)
                    
                    if volumes is not None and len(volumes) >= 10:
                        vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-10:])
                        volume_ratios.append(vol_ratio)
                        
                except:
                    continue
            
            if len(returns_5d) >= 3:
                avg_return_5d = np.mean(returns_5d)
                avg_return_10d = np.mean(returns_10d) if returns_10d else 0
                avg_volume_ratio = np.mean(volume_ratios) if volume_ratios else 1.0
                
                # 综合评分
                score = 0
                
                # 短期表现 (50%)
                if avg_return_5d > 0.03:
                    score += 0.5
                elif avg_return_5d > 0.01:
                    score += 0.3
                elif avg_return_5d > 0:
                    score += 0.1
                elif avg_return_5d > -0.02:
                    score += 0.05
                
                # 中期表现 (30%)
                if avg_return_10d > 0.05:
                    score += 0.3
                elif avg_return_10d > 0.02:
                    score += 0.2
                elif avg_return_10d > 0:
                    score += 0.1
                
                # 成交量确认 (20%)
                if avg_volume_ratio > 1.3:
                    score += 0.2
                elif avg_volume_ratio > 1.1:
                    score += 0.15
                elif avg_volume_ratio > 0.9:
                    score += 0.1
                
                sector_scores.append((sector_name, score))
                
        except Exception as e:
            if context.debug:
                print(f"分析行业 {sector_name} 时出错: {e}")
            continue
    
    # 按评分排序
    sector_scores.sort(key=lambda x: x[1], reverse=True)
    return sector_scores

def get_sector_stocks(sector_name, context):
    """获取行业股票列表"""
    instruments = all_instruments('CS')
    sector_stocks = []
    
    sector_keywords = context.sectors.get(sector_name, [])
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        
        # 跳过特殊股票
        symbol = stock_info.get('symbol', '')
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
        
        # 根据行业分类（简化版本，实际应该使用更准确的行业分类）
        industry = stock_info.get('industry_name', '')
        sector_type = stock_info.get('sector_code_name', '')
        
        # 检查是否属于目标行业
        belongs_to_sector = False
        for keyword in sector_keywords:
            if keyword in industry or keyword in sector_type:
                belongs_to_sector = True
                break
        
        if belongs_to_sector:
            sector_stocks.append(stock)
    
    return sector_stocks

def select_sector_stocks(context, bar_dict, sector_name, target_weight, current_date):
    """为特定行业选择股票"""
    sector_stocks = get_sector_stocks(sector_name, context)
    
    if not sector_stocks:
        if context.debug:
            print(f"{current_date} 行业 {sector_name} 无可选股票")
        return
    
    # 筛选并评分股票
    candidates = []
    
    for stock in sector_stocks[:50]:  # 只检查前50只
        try:
            if not basic_stock_filter(stock, bar_dict, context):
                continue
            
            score = calculate_stock_score(stock, context, sector_name)
            if score > 0.6:  # 评分阈值
                candidates.append((stock, score))
                
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 行业 {sector_name} 无合格股票")
        return
    
    # 按评分排序并选择
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_stocks = candidates[:context.stocks_per_sector]
    
    # 检查当前该行业持仓
    current_sector_stocks = []
    for stock, pos_info in context.positions.items():
        if pos_info['sector'] == sector_name:
            current_sector_stocks.append(stock)
    
    # 更新持仓
    target_stocks = set(stock for stock, _ in selected_stocks)
    current_stocks = set(current_sector_stocks)
    
    # 卖出不在目标列表的股票
    for stock in current_stocks - target_stocks:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 行业轮动卖出 {stock}")
    
    # 买入目标股票
    for stock, score in selected_stocks:
        order_target_percent(stock, target_weight)
        
        if stock not in context.positions:
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'sector': sector_name,
                'score': score
            }
        
        if context.debug:
            print(f"{current_date} 买入 {stock} ({sector_name}), 权重: {target_weight:.1%}, 评分: {score:.2f}")

def basic_stock_filter(stock, bar_dict, context):
    """基础股票筛选"""
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
        
        # 避免连续大跌的股票
        recent_decline = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        if recent_decline < -0.15:
            return False
        
        # 避免成交量异常萎缩
        vol_recent = np.mean(volumes[-3:])
        vol_base = np.mean(volumes[-10:])
        if vol_recent < vol_base * 0.5:
            return False
        
        return True
    
    except:
        return False

def calculate_stock_score(stock, context, sector_name):
    """计算股票评分"""
    try:
        closes = history_bars(stock, 30, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 20, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 20:
            return 0
        
        current = closes[-1]
        score = 0
        
        # 1. 均值回归机会 (40%)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        
        # 相对位置
        position_vs_ma5 = (current / ma5 - 1)
        position_vs_ma10 = (current / ma10 - 1)
        position_vs_ma20 = (current / ma20 - 1)
        
        # 寻找超跌后的反弹机会
        if -0.08 <= position_vs_ma5 <= -0.02 and position_vs_ma10 > -0.05:
            score += 0.4
        elif -0.05 <= position_vs_ma5 <= 0.02 and position_vs_ma20 > 0:
            score += 0.3
        elif position_vs_ma5 > 0 and position_vs_ma10 > 0:
            score += 0.2
        
        # 2. 短期动量 (30%)
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        
        if momentum_3d > 0.02 and momentum_5d > 0:
            score += 0.3
        elif momentum_3d > 0 and momentum_5d > -0.02:
            score += 0.2
        elif momentum_3d > -0.02:
            score += 0.1
        
        # 3. 成交量 (20%)
        if len(volumes) >= 10:
            vol_5 = np.mean(volumes[-5:])
            vol_10 = np.mean(volumes[-10:])
            vol_ratio = vol_5 / vol_10 if vol_10 > 0 else 1
            
            if vol_ratio > 1.2:
                score += 0.2
            elif vol_ratio > 1.0:
                score += 0.15
            elif vol_ratio > 0.8:
                score += 0.1
        
        # 4. 技术形态 (10%)
        if len(closes) >= 15:
            volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
            if 0.02 <= volatility <= 0.06:  # 适度波动
                score += 0.1
            elif volatility <= 0.08:
                score += 0.05
        
        return min(score, 1.0)
    
    except:
        return 0