# -*- coding: utf-8 -*-
"""
策略性能分析工具
================
用于分析和比较不同策略的回测表现
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def analyze_strategy_results():
    """分析策略回测结果"""
    
    print("=" * 60)
    print("策略表现分析报告 (2025-01-01 to 2025-07-01)")
    print("=" * 60)
    
    # 策略文件列表
    strategies = [
        "ai_tech_momentum_strategy.py",
        "multi_theme_rotation_strategy.py", 
        "high_frequency_swing_strategy.py",
        "sector_rotation_strategy.py",
        "trend_breakout_strategy.py"
    ]
    
    performance_summary = []
    
    for strategy in strategies:
        print(f"\n分析策略: {strategy}")
        print("-" * 40)
        
        # 这里需要用户手动输入每个策略的关键指标
        # 因为RQAlpha的结果通常在命令行输出中
        strategy_name = strategy.replace('.py', '').replace('_', ' ').title()
        
        print(f"策略名称: {strategy_name}")
        print("请手动输入以下指标 (从RQAlpha输出中获取):")
        
        try:
            total_return = float(input(f"  总收益率 (如: 0.15 表示15%): ") or "0")
            annual_return = float(input(f"  年化收益率 (如: 0.25 表示25%): ") or "0") 
            max_drawdown = float(input(f"  最大回撤 (如: -0.08 表示-8%): ") or "0")
            sharpe_ratio = float(input(f"  夏普比率 (如: 1.2): ") or "0")
            win_rate = float(input(f"  胜率 (如: 0.6 表示60%): ") or "0")
            
            performance_summary.append({
                'strategy': strategy_name,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_score': calculate_profit_score(total_return, max_drawdown, sharpe_ratio)
            })
            
        except KeyboardInterrupt:
            print("\n分析中断")
            break
        except:
            print(f"跳过策略 {strategy}")
            continue
    
    # 生成报告
    if performance_summary:
        generate_performance_report(performance_summary)
        recommend_strategies(performance_summary)
    
    return performance_summary

def calculate_profit_score(total_return, max_drawdown, sharpe_ratio):
    """计算策略盈利评分"""
    # 综合评分公式
    return_score = min(total_return * 4, 1.0)  # 收益率权重
    drawdown_score = max(0, 1 + max_drawdown * 5)  # 回撤惩罚
    sharpe_score = min(sharpe_ratio / 2, 1.0)  # 夏普比率权重
    
    total_score = (return_score * 0.5 + drawdown_score * 0.3 + sharpe_score * 0.2)
    return round(total_score * 100, 1)

def generate_performance_report(performance_data):
    """生成性能报告"""
    df = pd.DataFrame(performance_data)
    df = df.sort_values('profit_score', ascending=False)
    
    print("\n" + "=" * 80)
    print("策略排名表 (按盈利评分排序)")
    print("=" * 80)
    
    print(f"{'排名':<4} {'策略名称':<25} {'总收益率':<10} {'年化收益':<10} {'最大回撤':<10} {'夏普比率':<10} {'盈利评分':<10}")
    print("-" * 80)
    
    for i, row in df.iterrows():
        rank = df.index.get_loc(i) + 1
        print(f"{rank:<4} {row['strategy']:<25} {row['total_return']:>8.1%} {row['annual_return']:>8.1%} "
              f"{row['max_drawdown']:>8.1%} {row['sharpe_ratio']:>8.2f} {row['profit_score']:>8.1f}")

def recommend_strategies(performance_data):
    """推荐策略并标记需要删除的策略"""
    df = pd.DataFrame(performance_data)
    df = df.sort_values('profit_score', ascending=False)
    
    print("\n" + "=" * 60)
    print("策略建议")
    print("=" * 60)
    
    # 设定盈利阈值
    profit_threshold = 30.0  # 盈利评分阈值
    positive_return_threshold = 0.0  # 正收益阈值
    
    profitable_strategies = []
    unprofitable_strategies = []
    
    for _, row in df.iterrows():
        if row['total_return'] > positive_return_threshold and row['profit_score'] > profit_threshold:
            profitable_strategies.append(row['strategy'])
        else:
            unprofitable_strategies.append(row['strategy'])
    
    print(f"\n✅ 推荐保留的策略 ({len(profitable_strategies)}个):")
    for strategy in profitable_strategies:
        print(f"   - {strategy}")
    
    print(f"\n❌ 建议删除的策略 ({len(unprofitable_strategies)}个):")
    for strategy in unprofitable_strategies:
        print(f"   - {strategy}")
    
    if unprofitable_strategies:
        print(f"\n删除命令:")
        for strategy in unprofitable_strategies:
            filename = strategy.lower().replace(' ', '_') + '.py'
            print(f"   rm {filename}")
    
    return profitable_strategies, unprofitable_strategies

if __name__ == "__main__":
    print("策略性能分析工具")
    print("请确保已经运行完所有策略的回测")
    print("然后根据RQAlpha输出结果输入各项指标\n")
    
    results = analyze_strategy_results()