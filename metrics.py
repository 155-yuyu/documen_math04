"""
metrics.py - 6维指标计算模块
功能：计算成本、碳排、时延、QoS、新能源利用率、峰值净购电
"""

import numpy as np
from data_loader import N_REGIONS, REGIONS, T_TOTAL, T_MAIN, REGION_TO_IDX


def compute_metrics(all_sol, schedule_df, data, time_range=None,
                    renewable_override=None):
    """
    计算6维评价指标
    all_sol: get_all_solutions()返回的储能解字典
    schedule_df: 调度结果DataFrame
    data: 数据字典
    time_range: (t_start, t_end) 计算时段，默认0-2399
    renewable_override: 新能源覆盖数组(用于场景分析)
    """
    if time_range is None:
        time_range = (0, T_MAIN)

    t_start, t_end = time_range

    # ========== F1. 系统运行成本 ==========
    cost = 0.0
    for ri in range(N_REGIONS):
        if ri not in all_sol:
            continue
        sol = all_sol[ri]
        cost += np.sum(
            sol['p_buy'][t_start:t_end] * data['electricity_price'][ri, t_start:t_end] -
            sol['p_sell'][t_start:t_end] * data['sell_price'][ri, t_start:t_end])

    # ========== F2. 碳排放 ==========
    carbon = 0.0
    for ri in range(N_REGIONS):
        if ri not in all_sol:
            continue
        carbon += np.sum(
            all_sol[ri]['p_buy'][t_start:t_end] * data['carbon_intensity'][ri, t_start:t_end])

    # ========== F3. 加权网络时延 ==========
    latency_matrix = data['latency_matrix']
    source_idx_arr = np.array([REGION_TO_IDX[r] for r in schedule_df['SourceRegion'].values])
    exec_idx_arr = np.array([REGION_TO_IDX[r] for r in schedule_df['ExecRegion'].values])
    gpu_demand_arr = schedule_df['GPU_Demand'].values
    latencies = latency_matrix[source_idx_arr, exec_idx_arr]
    total_latency = np.sum(latencies * gpu_demand_arr)
    avg_latency = total_latency / len(schedule_df) if len(schedule_df) > 0 else 0
    n_migrated = int(np.sum(source_idx_arr != exec_idx_arr))

    # ========== F4. 服务质量 (按时完成率) ==========
    if 'OnTime' in schedule_df.columns:
        on_time = int(schedule_df['OnTime'].sum())
    else:
        on_time = int((schedule_df['FinishHour'] <= schedule_df['LatestFinishHour']).sum())
    n_total = len(schedule_df)
    qos_rate = on_time / n_total if n_total > 0 else 0

    # ========== F5. 新能源利用率 ==========
    renewable_used = 0.0
    renewable_total = 0.0
    for ri in range(N_REGIONS):
        if ri not in all_sol:
            continue
        sol = all_sol[ri]
        avail_ren = renewable_override[ri, :] if renewable_override is not None else data['available_renewable'][ri, :]
        renewable_used += np.sum(
            sol['p_ren_dir'][t_start:t_end] +
            sol['p_ren_ch'][t_start:t_end] +
            sol['p_ren_sell'][t_start:t_end])
        renewable_total += np.sum(avail_ren[t_start:t_end])
    renewable_utilization = renewable_used / renewable_total if renewable_total > 0 else 0

    # ========== F6. 区域峰值净购电 ==========
    net_import_total = np.zeros(t_end - t_start)
    for ri in range(N_REGIONS):
        if ri not in all_sol:
            continue
        net_import_total += all_sol[ri]['p_net'][t_start:t_end]
    peak_net_import = np.max(net_import_total) if len(net_import_total) > 0 else 0

    # 负荷波动
    fluctuation = np.sum(np.abs(np.diff(net_import_total))) if len(net_import_total) > 1 else 0

    return {
        'total_cost': cost,
        'total_carbon': carbon,
        'total_latency': total_latency,
        'avg_latency': avg_latency,
        'n_migrated': n_migrated,
        'migration_rate': n_migrated / n_total if n_total > 0 else 0,
        'qos_rate': qos_rate,
        'on_time': on_time,
        'n_total': n_total,
        'renewable_utilization': renewable_utilization,
        'peak_net_import': peak_net_import,
        'fluctuation': fluctuation,
    }


def compute_region_metrics(all_sol, data, time_range=None, renewable_override=None):
    """计算分区域指标"""
    if time_range is None:
        time_range = (0, T_MAIN)

    t_start, t_end = time_range
    results = {}

    for ri, r in enumerate(REGIONS):
        if ri not in all_sol:
            continue
        sol = all_sol[ri]
        avail_ren = renewable_override[ri, :] if renewable_override is not None else data['available_renewable'][ri, :]

        cost = np.sum(
            sol['p_buy'][t_start:t_end] * data['electricity_price'][ri, t_start:t_end] -
            sol['p_sell'][t_start:t_end] * data['sell_price'][ri, t_start:t_end])
        carbon = np.sum(
            sol['p_buy'][t_start:t_end] * data['carbon_intensity'][ri, t_start:t_end])
        ren_used = np.sum(
            sol['p_ren_dir'][t_start:t_end] +
            sol['p_ren_ch'][t_start:t_end] +
            sol['p_ren_sell'][t_start:t_end])
        ren_total = np.sum(avail_ren[t_start:t_end])
        peak_net = np.max(sol['p_net'][t_start:t_end])

        results[r] = {
            'cost': cost,
            'carbon': carbon,
            'renewable_utilization': ren_used / ren_total if ren_total > 0 else 0,
            'peak_net_import': peak_net,
            'avg_facility_load': np.mean(sol['facility_load'][t_start:t_end]),
        }

    return results


def compute_baseline_metrics(data, time_range=None):
    """计算基准策略指标（无任务迁移+固定储能）"""
    if time_range is None:
        time_range = (0, T_MAIN)
    t_start, t_end = time_range

    cost = np.sum(
        data['baseline_grid_buy'][:, t_start:t_end] * data['electricity_price'][:, t_start:t_end] -
        data['baseline_grid_sell'][:, t_start:t_end] * data['sell_price'][:, t_start:t_end])
    carbon = np.sum(
        data['baseline_grid_buy'][:, t_start:t_end] * data['carbon_intensity'][:, t_start:t_end])
    net_import = data['baseline_net_import'][:, t_start:t_end]
    peak_net = np.max(np.sum(net_import, axis=0))
    total_net = np.sum(net_import, axis=0)
    fluctuation = np.sum(np.abs(np.diff(total_net)))
    ren_used = np.sum(
        data['baseline_used_renewable'][:, t_start:t_end] +
        data['baseline_renewable_charge'][:, t_start:t_end] +
        data['baseline_grid_sell'][:, t_start:t_end])
    ren_total = np.sum(data['available_renewable'][:, t_start:t_end])

    return {
        'total_cost': cost,
        'total_carbon': carbon,
        'peak_net_import': peak_net,
        'fluctuation': fluctuation,
        'renewable_utilization': ren_used / ren_total if ren_total > 0 else 0,
        'total_latency': 0,
        'avg_latency': 0,
        'n_migrated': 0,
        'migration_rate': 0,
        'qos_rate': 1.0,
        'on_time': 50000,
        'n_total': 50000,
    }


def compare_metrics(baseline_metrics, optimized_metrics):
    """对比基准与优化后的指标"""
    comparison = {}
    for key in ['total_cost', 'total_carbon', 'total_latency',
                'renewable_utilization', 'peak_net_import', 'qos_rate']:
        b = baseline_metrics.get(key, 0)
        o = optimized_metrics.get(key, 0)
        if b != 0:
            change = (o - b) / abs(b) * 100
        else:
            change = 0
        comparison[key] = {
            'baseline': b,
            'optimized': o,
            'change_pct': change,
        }
    comparison['migration'] = {
        'n_migrated': optimized_metrics.get('n_migrated', 0),
        'migration_rate': optimized_metrics.get('migration_rate', 0),
    }
    return comparison
