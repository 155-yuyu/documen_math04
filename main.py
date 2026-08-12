"""
main.py - 主程序
流程：基准 → 优化 → Pareto → 场景分析 → 保存结果
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加当前目录到path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_all_data, prepare_task_arrays, REGIONS, N_REGIONS, T_TOTAL, T_MAIN
)
from task_scheduler import StorageAwareScheduler
from storage_optimizer import solve_all_regions, solve_all_weighted, get_all_solutions
from metrics import compute_metrics, compute_baseline_metrics, compute_region_metrics, compare_metrics
import scenario_analysis as sa
import visualization as viz

DATA_DIR = r'D:\数学建模\26数学建模暑期模拟训练\第二次训练题目（三选一）\C题 面向算电协同的多目标调度优化研究'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_实验结果')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6组Pareto权重
PARETO_WEIGHTS = [
    ('成本优先', (0.70, 0.10, 0.10, 0.10)),
    ('碳排优先', (0.10, 0.70, 0.10, 0.10)),
    ('时延优先', (0.10, 0.10, 0.70, 0.10)),
    ('新能源优先', (0.10, 0.10, 0.10, 0.70)),
    ('均衡策略', (0.25, 0.25, 0.25, 0.25)),
    ('成本-碳排均衡', (0.40, 0.30, 0.15, 0.15)),
]


def get_facility_load(scheduler, data):
    """从调度器获取设施负荷"""
    ai_load = scheduler.get_ai_it_load()
    it_load = data['nonai_it_load'] + ai_load
    return it_load * data['pue'][:, None]


def run_baseline(data, tasks):
    """运行基准策略"""
    print("\n" + "="*60)
    print("  基准策略 (无任务迁移 + 固定储能)")
    print("="*60)

    # 基准调度
    scheduler = StorageAwareScheduler(data, tasks)
    schedule_df = scheduler.run(mode='baseline')
    facility_load = get_facility_load(scheduler, data)

    # 基准储能 (使用数据中的基准值)
    baseline_m = compute_baseline_metrics(data)
    print(f"\n基准指标:")
    print(f"  成本: {baseline_m['total_cost']/1e4:.2f} 万元")
    print(f"  碳排: {baseline_m['total_carbon']:.0f} tCO2")
    print(f"  峰值净购电: {baseline_m['peak_net_import']:.1f} MW")
    print(f"  新能源利用率: {baseline_m['renewable_utilization']*100:.1f}%")

    return baseline_m, schedule_df


def run_optimized(data, tasks, weights, label):
    """运行单组权重的优化策略"""
    print(f"\n--- 策略: {label} (权重={weights}) ---")
    scheduler = StorageAwareScheduler(data, tasks, weights=weights)
    schedule_df = scheduler.run(mode='optimized')
    facility_load = get_facility_load(scheduler, data)

    # 储能优化 (成本最优)
    print("  储能优化 (成本最优)...")
    lp_cost = solve_all_regions(data, facility_load, objective='cost')
    sol_cost = get_all_solutions(data, facility_load, lp_cost)
    m_cost = compute_metrics(sol_cost, schedule_df, data)

    # 储能优化 (碳排最优)
    print("  储能优化 (碳排最优)...")
    lp_carbon = solve_all_regions(data, facility_load, objective='carbon')
    sol_carbon = get_all_solutions(data, facility_load, lp_carbon)
    m_carbon = compute_metrics(sol_carbon, schedule_df, data)

    # 选择碳排更优的解
    if m_carbon['total_carbon'] < m_cost['total_carbon']:
        all_sol = sol_carbon
        m = m_carbon
    else:
        all_sol = sol_cost
        m = m_cost

    m['label'] = label
    print(f"  结果: 成本={m['total_cost']/1e4:.2f}万元, 碳排={m['total_carbon']:.0f}tCO2, "
          f"利用率={m['renewable_utilization']*100:.1f}%, 峰值={m['peak_net_import']:.1f}MW, "
          f"迁移率={m['migration_rate']*100:.1f}%")

    return schedule_df, all_sol, m, facility_load


def run_pareto_analysis(data, tasks):
    """Pareto权重扫描"""
    print("\n" + "="*60)
    print("  Pareto前沿分析 (6组权重)")
    print("="*60)

    pareto_results = []
    best_all_sol = None
    best_m = None
    best_label = None
    best_schedule = None
    best_facility = None

    for label, weights in PARETO_WEIGHTS:
        schedule_df, all_sol, m, facility_load = run_optimized(data, tasks, weights, label)
        pareto_results.append(m)

        # 保存均衡策略的解用于后续分析
        if label == '均衡策略':
            best_all_sol = all_sol
            best_m = m
            best_label = label
            best_schedule = schedule_df
            best_facility = facility_load

    # 如果没有均衡策略，取最后一个
    if best_all_sol is None:
        best_all_sol = all_sol
        best_m = m
        best_label = label
        best_schedule = schedule_df
        best_facility = facility_load

    return pareto_results, best_all_sol, best_m, best_schedule, best_facility


def run_scenario_analysis(data, tasks, baseline_m):
    """场景分析"""
    print("\n" + "="*60)
    print("  场景分析")
    print("="*60)

    def facility_load_fn(scheduler):
        return get_facility_load(scheduler, data)

    # 1. 碳约束场景
    print("\n>>> 1. 碳约束场景 <<<")
    carbon_scenarios = sa.run_carbon_constraint_scenario(
        data, tasks, baseline_m, facility_load_fn)

    # 2. 电价机制场景
    print("\n>>> 2. 电价机制场景 <<<")
    price_scenarios = sa.run_price_mechanism_scenario(
        data, tasks, baseline_m, facility_load_fn)

    # 3. 新能源波动场景
    print("\n>>> 3. 新能源波动场景 <<<")
    renewable_scenarios = sa.run_renewable_scenario(
        data, tasks, baseline_m, facility_load_fn)

    return carbon_scenarios, price_scenarios, renewable_scenarios


def save_results(data, baseline_m, pareto_results, best_all_sol, best_m,
                best_schedule, best_facility, carbon_scenarios, price_scenarios,
                renewable_scenarios):
    """保存全部结果"""
    print("\n" + "="*60)
    print("  保存结果")
    print("="*60)

    # 1. 6维指标对比图
    viz.plot_6metric_comparison(baseline_m, best_m,
        os.path.join(OUTPUT_DIR, 'six_metric_comparison.png'))

    # 2. SOC轨迹图
    viz.plot_soc_trajectory(best_all_sol, data,
        os.path.join(OUTPUT_DIR, 'soc_trajectory.png'))

    # 3. 净购电曲线
    viz.plot_net_import(best_all_sol,
        os.path.join(OUTPUT_DIR, 'net_import.png'))

    # 4. 充放电策略
    viz.plot_charge_discharge(best_all_sol,
        os.path.join(OUTPUT_DIR, 'charge_discharge.png'))

    # 5. 分区域对比
    viz.plot_region_comparison(best_all_sol, data,
        os.path.join(OUTPUT_DIR, 'region_comparison.png'))

    # 6. Pareto前沿
    viz.plot_pareto(pareto_results,
        os.path.join(OUTPUT_DIR, 'pareto_frontier.png'))

    # 7. 任务分布
    viz.plot_task_distribution(best_schedule,
        os.path.join(OUTPUT_DIR, 'task_distribution.png'))

    # 8. 碳约束场景对比
    viz.plot_scenario_bar(carbon_scenarios,
        os.path.join(OUTPUT_DIR, 'carbon_scenario_bar.png'), baseline_m)
    viz.plot_scenario_radar(carbon_scenarios,
        os.path.join(OUTPUT_DIR, 'carbon_scenario_radar.png'), baseline_m)

    # 9. 电价场景对比
    viz.plot_scenario_bar(price_scenarios,
        os.path.join(OUTPUT_DIR, 'price_scenario_bar.png'), baseline_m)

    # 10. 新能源场景对比
    viz.plot_scenario_bar(renewable_scenarios,
        os.path.join(OUTPUT_DIR, 'renewable_scenario_bar.png'), baseline_m)

    # 保存Excel结果
    save_excel(data, baseline_m, pareto_results, best_m, best_schedule,
               carbon_scenarios, price_scenarios, renewable_scenarios)

    print(f"\n全部结果已保存至: {OUTPUT_DIR}")


def save_excel(data, baseline_m, pareto_results, best_m, schedule_df,
               carbon_scenarios, price_scenarios, renewable_scenarios):
    """保存Excel汇总结果"""
    filepath = os.path.join(OUTPUT_DIR, 'result.xlsx')
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet1: 6维指标对比
        comp = compare_metrics(baseline_m, best_m)
        rows = []
        for k, v in comp.items():
            if isinstance(v, dict):
                rows.append({
                    '指标': k,
                    '基准': v.get('baseline', ''),
                    '优化': v.get('optimized', ''),
                    '变化(%)': v.get('change_pct', ''),
                })
        pd.DataFrame(rows).to_excel(writer, sheet_name='6维指标对比', index=False)

        # Sheet2: Pareto结果
        pareto_rows = []
        for m in pareto_results:
            pareto_rows.append({
                '策略': m['label'],
                '成本(万元)': m['total_cost']/1e4,
                '碳排放(tCO2)': m['total_carbon'],
                '加权时延': m['total_latency'],
                '平均时延(ms)': m['avg_latency'],
                '迁移数': m['n_migrated'],
                '迁移率(%)': m['migration_rate']*100,
                'QoS(%)': m['qos_rate']*100,
                '新能源利用率(%)': m['renewable_utilization']*100,
                '峰值净购电(MW)': m['peak_net_import'],
                '负荷波动': m['fluctuation'],
            })
        pd.DataFrame(pareto_rows).to_excel(writer, sheet_name='Pareto结果', index=False)

        # Sheet3: 场景分析-碳约束
        scenario_rows = []
        for name, m in carbon_scenarios:
            scenario_rows.append({
                '场景': name,
                '成本(万元)': m['total_cost']/1e4,
                '碳排放(tCO2)': m['total_carbon'],
                '利用率(%)': m['renewable_utilization']*100,
                '峰值(MW)': m['peak_net_import'],
                '迁移率(%)': m['migration_rate']*100,
            })
        for name, m in price_scenarios:
            scenario_rows.append({
                '场景': f'电价-{name}',
                '成本(万元)': m['total_cost']/1e4,
                '碳排放(tCO2)': m['total_carbon'],
                '利用率(%)': m['renewable_utilization']*100,
                '峰值(MW)': m['peak_net_import'],
                '迁移率(%)': m['migration_rate']*100,
            })
        for name, m in renewable_scenarios:
            scenario_rows.append({
                '场景': f'新能源-{name}',
                '成本(万元)': m['total_cost']/1e4,
                '碳排放(tCO2)': m['total_carbon'],
                '利用率(%)': m['renewable_utilization']*100,
                '峰值(MW)': m['peak_net_import'],
                '迁移率(%)': m['migration_rate']*100,
            })
        pd.DataFrame(scenario_rows).to_excel(writer, sheet_name='场景分析', index=False)

        # Sheet4: 调度结果摘要
        sched_summary = schedule_df.groupby(['TaskType', 'ExecRegion']).agg(
            任务数=('TaskID', 'count'),
            总GPU=('GPU_Demand', 'sum'),
            平均时延=('MaxLatency_ms', 'mean')
        ).reset_index()
        sched_summary.to_excel(writer, sheet_name='调度摘要', index=False)

    print(f"[结果] result.xlsx 已保存")


def main():
    print("="*60)
    print("  问题四：多区域算-储-电协同优化模型")
    print("="*60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data = load_all_data(DATA_DIR)
    tasks = prepare_task_arrays(data['workload'])

    # 2. 基准策略
    print("\n[2/5] 基准策略...")
    baseline_m, baseline_schedule = run_baseline(data, tasks)

    # 3. Pareto分析
    print("\n[3/5] Pareto分析...")
    pareto_results, best_all_sol, best_m, best_schedule, best_facility = \
        run_pareto_analysis(data, tasks)

    # 4. 场景分析
    print("\n[4/5] 场景分析...")
    carbon_scenarios, price_scenarios, renewable_scenarios = \
        run_scenario_analysis(data, tasks, baseline_m)

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    save_results(data, baseline_m, pareto_results, best_all_sol, best_m,
                best_schedule, best_facility,
                carbon_scenarios, price_scenarios, renewable_scenarios)

    # 总结
    print("\n" + "="*60)
    print("  总结")
    print("="*60)
    comp = compare_metrics(baseline_m, best_m)
    print(f"基准 vs 均衡策略:")
    for k, v in comp.items():
        if isinstance(v, dict) and 'baseline' in v:
            print(f"  {k}: {v['baseline']:.2f} -> {v['optimized']:.2f} ({v['change_pct']:+.1f}%)")
    print(f"\n全部完成! 结果保存在: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
