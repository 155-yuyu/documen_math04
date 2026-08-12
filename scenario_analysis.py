"""
scenario_analysis.py - 场景分析模块
功能：碳约束、电价机制、新能源波动3类场景对比分析
"""

import numpy as np
from data_loader import REGIONS, N_REGIONS, T_TOTAL, T_MAIN
from task_scheduler import StorageAwareScheduler
from storage_optimizer import solve_all_regions, get_all_solutions
from metrics import compute_metrics, compute_baseline_metrics


def run_carbon_constraint_scenario(data, tasks, baseline_metrics, facility_load_fn,
                                    weights=(0.25, 0.25, 0.25, 0.25)):
    """
    碳约束场景分析
    facility_load_fn: 接收(scheduler)返回facility_load [6, 2407]的函数
    返回: [(场景名, metrics_dict), ...]
    """
    results = []
    baseline_carbon = baseline_metrics['total_carbon']
    scenarios = [
        ('无碳约束', None),
        ('中等碳约束(50%)', baseline_carbon * 0.5),
        ('严格碳约束(10%)', baseline_carbon * 0.1),
        ('零碳约束', 1.0),  # 近零
    ]

    for name, carbon_limit in scenarios:
        print(f"\n  === 场景: {name} ===")
        # 使用碳排优先权重模拟不同约束
        if carbon_limit is None:
            w = (0.70, 0.10, 0.10, 0.10)  # 成本优先
        elif carbon_limit <= 1.0:
            w = (0.10, 0.70, 0.10, 0.10)  # 碳排优先
        elif carbon_limit <= baseline_carbon * 0.1:
            w = (0.10, 0.70, 0.10, 0.10)
        else:
            w = (0.40, 0.30, 0.15, 0.15)  # 均衡

        scheduler = StorageAwareScheduler(data, tasks, weights=w)
        schedule_df = scheduler.run(mode='optimized')
        facility_load = facility_load_fn(scheduler)

        # 储能优化
        lp_results = solve_all_regions(data, facility_load, objective='cost')
        all_sol = get_all_solutions(data, facility_load, lp_results)

        # 用碳排优先目标再优化一次
        lp_results_c = solve_all_regions(data, facility_load, objective='carbon')
        all_sol_c = get_all_solutions(data, facility_load, lp_results_c)

        # 取碳排更优的解
        m_cost = compute_metrics(all_sol, schedule_df, data)
        m_carbon = compute_metrics(all_sol_c, schedule_df, data)
        if m_carbon['total_carbon'] < m_cost['total_carbon']:
            m = m_carbon
        else:
            m = m_cost

        results.append((name, m))
        print(f"    成本: {m['total_cost']/1e4:.2f}万元, 碳排: {m['total_carbon']:.0f}tCO2, "
              f"利用率: {m['renewable_utilization']*100:.1f}%, 峰值: {m['peak_net_import']:.1f}MW")

    return results


def run_price_mechanism_scenario(data, tasks, baseline_metrics, facility_load_fn,
                                  weights=(0.25, 0.25, 0.25, 0.25)):
    """
    电价机制场景分析
    """
    results = []
    mean_prices = np.mean(data['electricity_price'], axis=1)  # 各区域均价

    scenarios = [
        ('基准TOU', None, None),
        ('平价机制', mean_prices[:, None] * np.ones_like(data['electricity_price']), None),
        ('极端峰谷', None, 'extreme'),
    ]

    for name, price_override, mode in scenarios:
        print(f"\n  === 场景: {name} ===")
        scheduler = StorageAwareScheduler(data, tasks, weights=weights)
        schedule_df = scheduler.run(mode='optimized')
        facility_load = facility_load_fn(scheduler)

        if mode == 'extreme':
            # 极端峰谷：峰谷价比扩大2倍
            mean_p = np.mean(data['electricity_price'], axis=1)
            dev = data['electricity_price'] - mean_p[:, None]
            price_override = mean_p[:, None] + dev * 2.0
            price_override = np.maximum(price_override, 0)

        lp_results = solve_all_regions(data, facility_load, objective='cost')
        all_sol = get_all_solutions(data, facility_load, lp_results)

        # 如果有价格覆盖，需要重新计算成本
        m = compute_metrics(all_sol, schedule_df, data)
        if price_override is not None:
            # 用覆盖价格重新计算成本
            cost = 0.0
            for ri in range(N_REGIONS):
                if ri not in all_sol:
                    continue
                cost += np.sum(
                    all_sol[ri]['p_buy'][:T_MAIN] * price_override[ri, :T_MAIN] -
                    all_sol[ri]['p_sell'][:T_MAIN] * data['sell_price'][ri, :T_MAIN])
            m['total_cost'] = cost

        results.append((name, m))
        print(f"    成本: {m['total_cost']/1e4:.2f}万元, 碳排: {m['total_carbon']:.0f}tCO2, "
              f"利用率: {m['renewable_utilization']*100:.1f}%")

    return results


def run_renewable_scenario(data, tasks, baseline_metrics, facility_load_fn,
                           weights=(0.25, 0.25, 0.25, 0.25)):
    """
    新能源波动场景分析
    """
    results = []
    scenarios = [
        ('基准新能源', 1.0),
        ('高新能源(+20%)', 1.2),
        ('低新能源(-20%)', 0.8),
    ]

    for name, factor in scenarios:
        print(f"\n  === 场景: {name} ===")
        renewable_override = data['available_renewable'] * factor

        scheduler = StorageAwareScheduler(data, tasks, weights=weights,
                                          renewable_factor=factor)
        schedule_df = scheduler.run(mode='optimized')
        facility_load = facility_load_fn(scheduler)

        lp_results = solve_all_regions(data, facility_load, objective='cost',
                                        renewable_override=renewable_override)
        all_sol = get_all_solutions(data, facility_load, lp_results,
                                    renewable_override=renewable_override)

        m = compute_metrics(all_sol, schedule_df, data, renewable_override=renewable_override)
        results.append((name, m))
        print(f"    成本: {m['total_cost']/1e4:.2f}万元, 碳排: {m['total_carbon']:.0f}tCO2, "
              f"利用率: {m['renewable_utilization']*100:.1f}%, 峰值: {m['peak_net_import']:.1f}MW")

    return results
