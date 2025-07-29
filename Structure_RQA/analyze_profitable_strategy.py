# -*- coding: utf-8 -*-
"""
Strategy Performance Analysis
"""

from rqalpha.api import *
import numpy as np
import pandas as pd

def calculate_performance_metrics(context):
    """计算策略绩效指标"""
    if not hasattr(context, 'performance_tracker') or not context.performance_tracker:
        return {}
    
    df = pd.DataFrame(context.performance_tracker)
    
    # 计算收益率
    initial_nav = df['nav'].iloc[0]
    final_nav = df['nav'].iloc[-1]
    total_return = (final_nav / initial_nav - 1) * 100
    
    # 计算年化收益率
    days = len(df)
    annualized_return = ((final_nav / initial_nav) ** (252 / days) - 1) * 100
    
    # 计算最大回撤
    cumulative_returns = df['nav'] / initial_nav
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min() * 100
    
    # 计算胜率
    daily_returns = df['daily_return'].dropna()
    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
    
    # 计算夏普比率 (假设无风险利率为3%)
    risk_free_rate = 0.03
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # 计算波动率
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # 交易统计
    avg_positions = df['positions'].mean()
    max_positions = df['positions'].max()
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'avg_positions': avg_positions,
        'max_positions': max_positions,
        'trading_days': days
    }
    
    return metrics

def print_performance_report(metrics):
    """打印绩效报告"""
    print("\n" + "="*50)
    print("PROFITABLE BREAKOUT STRATEGY PERFORMANCE REPORT")
    print("="*50)
    print(f"回测期间: 2025-01-01 至 2025-07-01")
    print(f"交易天数: {metrics['trading_days']}")
    print("-"*50)
    print("收益指标:")
    print(f"  总收益率: {metrics['total_return']:.2f}%")
    print(f"  年化收益率: {metrics['annualized_return']:.2f}%")
    print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    print("-"*50)
    print("风险指标:")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  年化波动率: {metrics['volatility']:.2f}%")
    print(f"  胜率: {metrics['win_rate']:.2f}%")
    print("-"*50)
    print("交易统计:")
    print(f"  平均持仓数: {metrics['avg_positions']:.1f}")
    print(f"  最大持仓数: {metrics['max_positions']}")
    print("="*50)
    
    # 判断是否达到目标
    target_return = 25.0
    if metrics['annualized_return'] >= target_return:
        print(f"✅ 策略成功! 年化收益率 {metrics['annualized_return']:.2f}% 超过目标 {target_return}%")
    else:
        print(f"❌ 需要优化! 年化收益率 {metrics['annualized_return']:.2f}% 未达到目标 {target_return}%")
        gap = target_return - metrics['annualized_return']
        print(f"   距离目标还差: {gap:.2f}%")
    
    print("="*50)

# 如果直接运行此文件，则输出示例分析
if __name__ == "__main__":
    # 创建示例数据来演示
    print("Strategy Performance Analysis Module")
    print("This module provides tools to analyze trading strategy performance.")
    print("Use it in combination with backtest results from RQAlpha.")