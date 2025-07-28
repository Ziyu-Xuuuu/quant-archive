# -*- coding: utf-8 -*-
"""
多主题轮动策略 2025 v1.0
========================
针对2025年多个热门主题的轮动策略

核心逻辑：
1. 多主题覆盖 - AI科技、新能源、消费升级、医药创新、军工国防
2. 主题强度评估 - 动态识别最强势主题
3. 主题内选股 - 每个主题选择最优标的
4. 灵活轮动 - 根据市场热点快速切换
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
    context.max_themes = 3             # 最多3个主题
    context.stocks_per_theme = 2       # 每主题2只股票
    context.total_positions = 6        # 总计6只股票
    context.base_position = 0.92       # 高仓位运作
    context.single_position_limit = 0.20  # 单股最大20%
    
    # 主题定义
    context.themes = {
        'ai_tech': {
            'name': 'AI科技',
            'keywords': ['人工智能', '机器学习', '深度学习', '算法', '芯片', '半导体', 
                        '云计算', '大数据', '物联网', '5G', '软件开发', '信息技术',
                        '自动驾驶', '机器人', '智能制造', '区块链', '量子计算'],
            'weight': 0
        },
        'new_energy': {
            'name': '新能源',
            'keywords': ['新能源', '电动汽车', '锂电池', '光伏', '风电', '储能',
                        '充电桩', '氢能源', '核电', '清洁能源', '碳中和', '节能环保',
                        '电池', '太阳能', '风力发电', '电力设备'],
            'weight': 0
        },
        'consumption_upgrade': {
            'name': '消费升级', 
            'keywords': ['消费电子', '智能家居', '品牌服装', '高端食品', '化妆品',
                        '医美', '教育', '娱乐', '旅游', '体育', '健康', '养老',
                        '奢侈品', '精品咖啡', '新零售', '直播电商'],
            'weight': 0
        },
        'biotech_pharma': {
            'name': '医药创新',
            'keywords': ['生物医药', '创新药', '疫苗', '基因治疗', '细胞治疗',
                        '医疗器械', '诊断试剂', 'CRO', 'CDMO', '原料药',
                        '中药', '医疗服务', '互联网医疗', '精准医疗'],
            'weight': 0
        },
        'defense_military': {
            'name': '军工国防',
            'keywords': ['军工', '国防', '航空', '航天', '雷达', '导弹', '无人机',
                        '卫星', '通信设备', '电子对抗', '军用材料', '船舶',
                        '核工业', '兵器', '军民融合'],
            'weight': 0
        }
    }
    
    # 轮动参数
    context.rotation_period = 4        # 每4天评估一次
    context.theme_momentum_period = 8  # 主题动量计算周期
    context.min_theme_strength = 0.6   # 主题最小强度阈值
    
    # 风控参数
    context.stop_loss = 0.07           # 7%止损
    context.take_profit = 0.18         # 18%止盈
    context.max_hold_days = 15         # 最大持有15天
    
    # 筛选条件
    context.min_price = 6.0
    context.max_price = 120.0
    context.min_market_cap = 40e8      # 最小市值40亿
    context.min_volume_value = 60e6    # 日成交额6000万+
    
    # 状态记录
    context.positions = {}
    context.theme_stock_pools = {}
    context.theme_performance = {}
    context.last_rotation = None
    context.performance_tracker = []
    context.debug = True
    
    # 初始化主题股票池
    initialize_theme_pools(context)
    print(f"多主题轮动策略初始化完成")
    for theme, info in context.themes.items():
        pool_size = len(context.theme_stock_pools.get(theme, []))
        print(f"  {info['name']}: {pool_size}只股票")

def initialize_theme_pools(context):
    """初始化各主题股票池"""
    instruments = all_instruments('CS')
    context.theme_stock_pools = {theme: [] for theme in context.themes.keys()}
    
    for _, stock_info in instruments.iterrows():
        stock = stock_info['order_book_id']
        symbol = stock_info.get('symbol', '')
        company_name = stock_info.get('display_name', '')
        industry = stock_info.get('industry_name', '')
        
        # 跳过特殊股票
        if any(x in symbol for x in ['ST', '*ST', '退']):
            continue
        
        # 检查股票属于哪个主题
        text_to_check = f"{company_name} {industry}".lower()
        
        for theme_id, theme_info in context.themes.items():
            for keyword in theme_info['keywords']:
                if keyword in text_to_check:
                    context.theme_stock_pools[theme_id].append(stock)
                    break

def handle_bar(context, bar_dict):
    """主策略逻辑"""
    current_date = context.now.date()
    
    # 记录表现
    nav = context.portfolio.total_value
    context.performance_tracker.append({
        'date': current_date,
        'nav': nav,
        'positions': len([p for p in context.portfolio.positions.values() if p.quantity > 0]),
        'themes': list(set(pos['theme'] for pos in context.positions.values()))
    })
    
    # 管理现有持仓
    manage_theme_positions(context, bar_dict, current_date)
    
    # 定期主题轮动
    if should_rotate_themes(context, current_date):
        rotate_themes(context, bar_dict, current_date)

def manage_theme_positions(context, bar_dict, current_date):
    """管理主题持仓"""
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
        theme = pos_info['theme']
        
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
        elif check_theme_weakness(theme, context):
            should_close, reason = True, f"主题转弱({context.themes[theme]['name']})，收益: {return_pct:.2%}"
        elif check_stock_momentum_loss(stock, context):
            should_close, reason = True, f"个股动量衰竭，收益: {return_pct:.2%}"
        
        if should_close:
            to_close.append((stock, reason))
    
    # 执行平仓
    for stock, reason in to_close:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 平仓 {stock}: {reason}")

def check_theme_weakness(theme, context):
    """检查主题是否转弱"""
    try:
        theme_stocks = context.theme_stock_pools.get(theme, [])
        if len(theme_stocks) < 5:
            return False
        
        # 计算主题平均表现
        theme_returns = []
        for stock in theme_stocks[:20]:  # 取前20只
            try:
                closes = history_bars(stock, 6, '1d', 'close', skip_suspended=True, include_now=True)
                if closes is not None and len(closes) >= 6:
                    ret = (closes[-1] / closes[-6] - 1)
                    theme_returns.append(ret)
            except:
                continue
        
        if len(theme_returns) >= 5:
            avg_return = np.mean(theme_returns)
            return avg_return < -0.04  # 主题平均跌幅超过4%
        
        return False
    except:
        return False

def check_stock_momentum_loss(stock, context):
    """检查个股动量是否衰竭"""
    try:
        closes = history_bars(stock, 8, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 8, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 8:
            return False
        
        # 连续3天下跌
        recent_changes = np.diff(closes[-4:])
        if np.sum(recent_changes < 0) >= 3:
            return True
        
        # 成交量大幅萎缩
        vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-8:])
        if vol_ratio < 0.6:
            return True
        
        return False
    except:
        return False

def should_rotate_themes(context, current_date):
    """判断是否需要主题轮动"""
    if context.last_rotation is None:
        return True
    
    days_since_rotation = (current_date - context.last_rotation).days
    return days_since_rotation >= context.rotation_period

def rotate_themes(context, bar_dict, current_date):
    """执行主题轮动"""
    if context.debug:
        print(f"{current_date} 开始主题轮动分析")
    
    # 分析各主题强度
    theme_strengths = analyze_theme_strengths(context)
    
    if not theme_strengths:
        if context.debug:
            print(f"{current_date} 无法获取主题强度数据")
        return
    
    # 选择最强的主题
    top_themes = theme_strengths[:context.max_themes]
    selected_themes = [theme for theme, strength in top_themes if strength >= context.min_theme_strength]
    
    if not selected_themes:
        if context.debug:
            print(f"{current_date} 没有符合条件的强势主题")
        return
    
    if context.debug:
        theme_names = [context.themes[theme]['name'] for theme in selected_themes]
        print(f"{current_date} 选择主题: {theme_names}")
    
    # 清理不在选择列表的主题持仓
    current_themes = set(pos['theme'] for pos in context.positions.values())
    themes_to_clear = current_themes - set(selected_themes)
    
    for theme in themes_to_clear:
        stocks_to_clear = [stock for stock, pos in context.positions.items() if pos['theme'] == theme]
        for stock in stocks_to_clear:
            order_target_percent(stock, 0)
            context.positions.pop(stock, None)
            if context.debug:
                print(f"{current_date} 清理主题 {context.themes[theme]['name']}: {stock}")
    
    # 为每个选中主题分配仓位
    theme_weight = context.base_position / len(selected_themes)
    stock_weight = theme_weight / context.stocks_per_theme
    
    for theme in selected_themes:
        select_theme_stocks(context, bar_dict, theme, stock_weight, current_date)
    
    context.last_rotation = current_date

def analyze_theme_strengths(context):
    """分析各主题强度"""
    theme_strengths = []
    
    for theme_id, theme_info in context.themes.items():
        try:
            strength = calculate_theme_strength(context, theme_id)
            if strength > 0:
                theme_strengths.append((theme_id, strength))
        except Exception as e:
            if context.debug:
                print(f"分析主题 {theme_info['name']} 时出错: {e}")
            continue
    
    # 按强度排序
    theme_strengths.sort(key=lambda x: x[1], reverse=True)
    return theme_strengths

def calculate_theme_strength(context, theme_id):
    """计算主题强度"""
    theme_stocks = context.theme_stock_pools.get(theme_id, [])
    if len(theme_stocks) < 3:
        return 0
    
    returns_3d = []
    returns_8d = []
    volume_ratios = []
    momentum_scores = []
    
    for stock in theme_stocks[:30]:  # 取前30只代表股票
        try:
            closes = history_bars(stock, 15, '1d', 'close', skip_suspended=True, include_now=True)
            volumes = history_bars(stock, 12, '1d', 'volume', skip_suspended=True, include_now=True)
            
            if closes is not None and len(closes) >= 10:
                ret_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
                ret_8d = (closes[-1] / closes[-9] - 1) if len(closes) >= 9 else 0
                returns_3d.append(ret_3d)
                returns_8d.append(ret_8d)
                
                # 动量评分
                momentum = calculate_single_momentum(closes)
                momentum_scores.append(momentum)
            
            if volumes is not None and len(volumes) >= 8:
                vol_ratio = np.mean(volumes[-3:]) / np.mean(volumes[-8:])
                volume_ratios.append(vol_ratio)
                
        except:
            continue
    
    if len(returns_3d) < 3:
        return 0
    
    # 综合评分
    strength = 0
    
    # 短期表现 (40%)
    avg_return_3d = np.mean(returns_3d)
    if avg_return_3d > 0.03:
        strength += 0.4
    elif avg_return_3d > 0.01:
        strength += 0.3
    elif avg_return_3d > 0:
        strength += 0.2
    elif avg_return_3d > -0.02:
        strength += 0.1
    
    # 中期表现 (30%)
    if returns_8d:
        avg_return_8d = np.mean(returns_8d)
        if avg_return_8d > 0.06:
            strength += 0.3
        elif avg_return_8d > 0.03:
            strength += 0.2
        elif avg_return_8d > 0:
            strength += 0.15
        elif avg_return_8d > -0.03:
            strength += 0.1
    
    # 成交量确认 (20%)
    if volume_ratios:
        avg_volume_ratio = np.mean(volume_ratios)
        if avg_volume_ratio > 1.4:
            strength += 0.2
        elif avg_volume_ratio > 1.2:
            strength += 0.15
        elif avg_volume_ratio > 1.0:
            strength += 0.1
    
    # 动量一致性 (10%)
    if momentum_scores:
        positive_momentum_ratio = sum(1 for m in momentum_scores if m > 0.5) / len(momentum_scores)
        strength += 0.1 * positive_momentum_ratio
    
    return min(strength, 1.0)

def calculate_single_momentum(closes):
    """计算单个股票的动量评分"""
    if len(closes) < 8:
        return 0
    
    score = 0
    current = closes[-1]
    
    # 短期动量
    momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
    if momentum_3d > 0.02:
        score += 0.4
    elif momentum_3d > 0:
        score += 0.2
    
    # 均线位置
    if len(closes) >= 8:
        ma5 = np.mean(closes[-5:])
        ma8 = np.mean(closes[-8:])
        
        if current > ma5 > ma8:
            score += 0.4
        elif current > ma5:
            score += 0.2
    
    # 趋势稳定性
    if len(closes) >= 6:
        recent_volatility = np.std(closes[-6:]) / np.mean(closes[-6:])
        if recent_volatility < 0.05:
            score += 0.2
        elif recent_volatility < 0.08:
            score += 0.1
    
    return min(score, 1.0)

def select_theme_stocks(context, bar_dict, theme_id, target_weight, current_date):
    """为特定主题选择股票"""
    theme_stocks = context.theme_stock_pools.get(theme_id, [])
    theme_name = context.themes[theme_id]['name']
    
    if not theme_stocks:
        if context.debug:
            print(f"{current_date} 主题 {theme_name} 无可选股票")
        return
    
    # 筛选并评分股票
    candidates = []
    
    for stock in theme_stocks[:50]:  # 只检查前50只
        try:
            if not basic_theme_stock_filter(stock, bar_dict, context):
                continue
            
            score = calculate_theme_stock_score(stock, context)
            if score > 0.6:  # 评分阈值
                candidates.append((stock, score))
                
        except:
            continue
    
    if not candidates:
        if context.debug:
            print(f"{current_date} 主题 {theme_name} 无合格股票")
        return
    
    # 按评分排序并选择
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_stocks = candidates[:context.stocks_per_theme]
    
    # 检查当前该主题持仓
    current_theme_stocks = [stock for stock, pos in context.positions.items() if pos['theme'] == theme_id]
    
    # 更新持仓
    target_stocks = set(stock for stock, _ in selected_stocks)
    current_stocks = set(current_theme_stocks)
    
    # 卖出不在目标列表的股票
    for stock in current_stocks - target_stocks:
        order_target_percent(stock, 0)
        context.positions.pop(stock, None)
        if context.debug:
            print(f"{current_date} 主题轮动卖出 {stock}")
    
    # 买入目标股票
    for stock, score in selected_stocks:
        order_target_percent(stock, target_weight)
        
        if stock not in context.positions:
            context.positions[stock] = {
                'entry_date': current_date,
                'entry_price': bar_dict[stock].close,
                'theme': theme_id,
                'score': score
            }
        
        if context.debug:
            print(f"{current_date} 买入 {stock} ({theme_name}), 权重: {target_weight:.1%}, 评分: {score:.2f}")

def basic_theme_stock_filter(stock, bar_dict, context):
    """基础主题股票筛选"""
    try:
        price = bar_dict[stock].close
        
        # 价格区间
        if not (context.min_price <= price <= context.max_price):
            return False
        
        # 获取数据
        closes = history_bars(stock, 25, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return False
        
        # 流动性检查
        avg_volume_value = np.mean(closes[-8:] * volumes[-8:])
        if avg_volume_value < context.min_volume_value:
            return False
        
        # 避免连续大跌
        recent_decline = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        if recent_decline < -0.12:
            return False
        
        return True
    
    except:
        return False

def calculate_theme_stock_score(stock, context):
    """计算主题股票评分"""
    try:
        closes = history_bars(stock, 20, '1d', 'close', skip_suspended=True, include_now=True)
        volumes = history_bars(stock, 15, '1d', 'volume', skip_suspended=True, include_now=True)
        
        if closes is None or volumes is None or len(closes) < 15:
            return 0
        
        current = closes[-1]
        score = 0
        
        # 1. 动量表现 (40%)
        momentum_3d = (closes[-1] / closes[-4] - 1) if len(closes) >= 4 else 0
        momentum_8d = (closes[-1] / closes[-9] - 1) if len(closes) >= 9 else 0
        
        if momentum_3d > 0.03 and momentum_8d > 0:
            score += 0.4
        elif momentum_3d > 0.01 and momentum_8d > -0.02:
            score += 0.3
        elif momentum_3d > 0:
            score += 0.2
        elif momentum_3d > -0.02:
            score += 0.1
        
        # 2. 技术形态 (30%)
        if len(closes) >= 15:
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma15 = np.mean(closes[-15:])
            
            if current > ma5 > ma10 > ma15:
                score += 0.3
            elif current > ma5 > ma10:
                score += 0.2
            elif current > ma5:
                score += 0.15
        
        # 3. 成交量 (20%)
        if len(volumes) >= 10:
            vol_recent = np.mean(volumes[-3:])
            vol_base = np.mean(volumes[-10:])
            vol_ratio = vol_recent / vol_base if vol_base > 0 else 1
            
            if vol_ratio > 1.5:
                score += 0.2
            elif vol_ratio > 1.2:
                score += 0.15
            elif vol_ratio > 1.0:
                score += 0.1
        
        # 4. 相对强度 (10%)
        if len(closes) >= 15:
            volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
            if 0.03 <= volatility <= 0.08:  # 适度波动
                score += 0.1
            elif volatility <= 0.10:
                score += 0.05
        
        return min(score, 1.0)
    
    except:
        return 0