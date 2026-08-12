"""
data_loader.py - 统一数据加载模块
功能：加载任务负载、电力参数、储能参数、GPU容量、网络时延全部数据
合并问题二(data_process.py) + 问题三(data_loader.py)的加载逻辑
"""

import pandas as pd
import numpy as np
import os
import math

# ==================== 全局常量 ====================
REGIONS = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}
N_REGIONS = 6
T_TOTAL = 2407  # 0-2406小时
T_MAIN = 2400   # 0-2399为主时域

TASK_TYPES = ['RealTimeInference', 'BatchInference', 'AITraining']
TASK_TYPE_TO_IDX = {t: i for i, t in enumerate(TASK_TYPES)}
GPU_POWER = {  # TaskType -> MW per GPU
    'AITraining': 0.16,
    'BatchInference': 0.10,
    'RealTimeInference': 0.08
}


def load_all_data(data_dir):
    """
    加载全部数据文件，返回数据字典
    data_dir: C题根目录路径
    """
    data = {}

    # ==================== 1. GPU_information.xlsx ====================
    gpu_info = pd.read_excel(
        os.path.join(data_dir, 'GPU_information.xlsx'),
        sheet_name='GPU中心基础情况')
    data['gpu_info'] = gpu_info

    data['total_gpu'] = np.array(
        [gpu_info.loc[gpu_info['Region'] == r, 'Total_GPU'].values[0] for r in REGIONS],
        dtype=np.float64)
    data['available_gpu'] = np.array(
        [gpu_info.loc[gpu_info['Region'] == r, 'Available_GPU'].values[0] for r in REGIONS],
        dtype=np.float64)
    data['max_it_power'] = np.array(
        [gpu_info.loc[gpu_info['Region'] == r, 'Max_IT_Power_MW'].values[0] for r in REGIONS],
        dtype=np.float64)
    data['pue'] = np.array(
        [gpu_info.loc[gpu_info['Region'] == r, 'PUE'].values[0] for r in REGIONS],
        dtype=np.float64)
    data['max_facility_power'] = np.array(
        [gpu_info.loc[gpu_info['Region'] == r, 'Max_Facility_Power_MW'].values[0] for r in REGIONS],
        dtype=np.float64)
    print(f"[GPU信息] 可用GPU: {data['available_gpu']}")

    # ==================== 2. network_latency.xlsx ====================
    latency_df = pd.read_excel(
        os.path.join(data_dir, 'network_latency.xlsx'),
        sheet_name='时延矩阵')
    latency_matrix = np.zeros((N_REGIONS, N_REGIONS))
    for i, from_r in enumerate(REGIONS):
        row = latency_df[latency_df['From\\To'] == from_r].iloc[0]
        for j, to_r in enumerate(REGIONS):
            latency_matrix[i, j] = row[to_r]
    data['latency_matrix'] = latency_matrix
    print(f"[网络时延] 矩阵加载完成")

    # ==================== 3. workload_trace.xlsx ====================
    workload = pd.read_excel(
        os.path.join(data_dir, 'workload_trace.xlsx'),
        sheet_name='Sheet1')
    data['workload'] = workload
    print(f"[工作负载] {len(workload)}条任务加载完成")

    # ==================== 4. power_mapping.xlsx ====================
    pm = pd.read_excel(
        os.path.join(data_dir, 'dataset', 'power_mapping.xlsx'),
        sheet_name='任务功率映射')
    for _, row in pm.iterrows():
        assert abs(row['GPU_Power_MW_per_EquivalentGPU'] - GPU_POWER[row['TaskType']]) < 1e-6, \
            f"功率映射不匹配: {row['TaskType']}"
    print(f"[功率映射] 验证通过: {GPU_POWER}")

    # ==================== 5. region_time_data.xlsx ====================
    rtd = pd.read_excel(
        os.path.join(data_dir, 'dataset', 'region_time_data.xlsx'),
        sheet_name='region_time_data')
    data['rtd'] = rtd

    def build_region_hour_array(col_name):
        arr = np.zeros((N_REGIONS, T_TOTAL))
        for ri, r in enumerate(REGIONS):
            subset = rtd[rtd['Region'] == r].sort_values('Hour')
            hours = subset['Hour'].values.astype(int)
            vals = subset[col_name].values
            arr[ri, hours] = vals
        return arr

    data['electricity_price'] = build_region_hour_array('ElectricityPrice_CNY_per_MWh')
    data['sell_price'] = build_region_hour_array('SellPrice_CNY_per_MWh')
    data['carbon_intensity'] = build_region_hour_array('CarbonIntensity_tCO2_per_MWh')
    data['available_renewable'] = build_region_hour_array('AvailableRenewable_MW')
    data['nonai_it_load'] = build_region_hour_array('NonAI_IT_Load_MW')
    data['baseline_ai_it_load'] = build_region_hour_array('Baseline_AI_IT_Load_MW')
    # 基准储能参数
    data['baseline_charge_power'] = build_region_hour_array('ChargePower_MW')
    data['baseline_discharge_power'] = build_region_hour_array('DischargePower_MW')
    data['baseline_renewable_charge'] = build_region_hour_array('RenewableCharge_MW')
    data['baseline_grid_buy'] = build_region_hour_array('GridPurchase_MW')
    data['baseline_grid_sell'] = build_region_hour_array('GridSell_MW')
    data['baseline_net_import'] = build_region_hour_array('NetGridImport_MW')
    data['baseline_used_renewable'] = build_region_hour_array('UsedRenewable_MW')
    data['baseline_curtailment'] = build_region_hour_array('Curtailment_MW')
    data['baseline_soc'] = build_region_hour_array('SOC_MWh')

    print(f"[区域时间数据] {len(rtd)}条记录加载完成")

    # ==================== 6. storage_information.xlsx ====================
    storage = pd.read_excel(
        os.path.join(data_dir, 'dataset', 'storage_information.xlsx'),
        sheet_name='storage_information')
    data['storage'] = storage

    data['storage_cap'] = np.array(
        [storage.loc[storage['Region'] == r, 'StorageCapacity_MWh'].values[0] for r in REGIONS])
    data['min_soc'] = np.array(
        [storage.loc[storage['Region'] == r, 'MinSOC_MWh'].values[0] for r in REGIONS])
    data['init_soc'] = np.array(
        [storage.loc[storage['Region'] == r, 'InitialSOC_MWh'].values[0] for r in REGIONS])
    data['max_charge'] = np.array(
        [storage.loc[storage['Region'] == r, 'MaxChargePower_MW'].values[0] for r in REGIONS])
    data['max_discharge'] = np.array(
        [storage.loc[storage['Region'] == r, 'MaxDischargePower_MW'].values[0] for r in REGIONS])
    data['eta_ch'] = np.array(
        [storage.loc[storage['Region'] == r, 'ChargeEfficiency'].values[0] for r in REGIONS])
    data['eta_dis'] = np.array(
        [storage.loc[storage['Region'] == r, 'DischargeEfficiency'].values[0] for r in REGIONS])
    data['max_grid_import'] = np.array(
        [storage.loc[storage['Region'] == r, 'MaxGridImport_MW'].values[0] for r in REGIONS])
    data['max_grid_export'] = np.array(
        [storage.loc[storage['Region'] == r, 'MaxGridExport_MW'].values[0] for r in REGIONS])
    data['sell_limit'] = np.array(
        [storage.loc[storage['Region'] == r, 'SellLimit_MW'].values[0] for r in REGIONS])

    print(f"[储能信息] 容量: {data['storage_cap']}")

    # ==================== 7. 预计算费率数组 ====================
    # cost_rate[r,t] = ElectricityPrice(r,t) * PUE(r)  [元/MWh per MW of IT load]
    data['cost_rate'] = data['electricity_price'] * data['pue'][:, None]
    data['carbon_rate'] = data['carbon_intensity'] * data['pue'][:, None]

    # 基准IT负荷和设施负荷
    data['baseline_it_load'] = data['baseline_ai_it_load'] + data['nonai_it_load']
    data['baseline_facility_load'] = data['baseline_it_load'] * data['pue'][:, None]

    print(f"[预处理] cost_rate/carbon_rate 计算完成")
    return data


def compute_overlap(duration_min):
    """
    计算任务的逐小时重叠系数
    例如 duration_min=280 -> overlap=[1.0, 1.0, 1.0, 1.0, 0.6667], num_hours=5
    """
    if duration_min <= 0:
        return np.array([0.0]), 0
    num_full = duration_min // 60
    partial_min = duration_min % 60
    if partial_min > 0:
        overlap = np.ones(num_full + 1)
        overlap[-1] = partial_min / 60.0
        return overlap, num_full + 1
    else:
        overlap = np.ones(num_full)
        return overlap, num_full


def get_duration_hours(duration_min):
    """返回任务占用的完整小时数（向上取整）"""
    return math.ceil(duration_min / 60.0)


def prepare_task_arrays(workload_df):
    """将工作负载DataFrame转换为numpy数组，加速访问"""
    n = len(workload_df)
    task_ids = workload_df['TaskID'].values
    task_types = workload_df['TaskType'].values
    arrival_hours = workload_df['ArrivalHour'].values.astype(np.int32)
    gpu_demands = workload_df['GPU_Demand'].values.astype(np.float64)
    durations_min = workload_df['EstimatedDuration_min'].values.astype(np.int32)
    source_regions = workload_df['SourceRegion'].values
    max_latencies = workload_df['MaxLatency_ms'].values.astype(np.int32)
    latest_finish = workload_df['LatestFinishHour'].values.astype(np.int32)

    source_region_idx = np.array([REGION_TO_IDX[r] for r in source_regions], dtype=np.int32)
    task_type_idx = np.array([TASK_TYPE_TO_IDX[t] for t in task_types], dtype=np.int32)
    task_power_per_gpu = np.array([GPU_POWER[t] for t in task_types], dtype=np.float64)
    task_it_power = gpu_demands * task_power_per_gpu
    duration_hours = np.ceil(durations_min / 60.0).astype(np.int32)

    return {
        'task_ids': task_ids,
        'task_types': task_types,
        'task_type_idx': task_type_idx,
        'arrival_hours': arrival_hours,
        'gpu_demands': gpu_demands,
        'durations_min': durations_min,
        'duration_hours': duration_hours,
        'source_region_idx': source_region_idx,
        'max_latencies': max_latencies,
        'latest_finish': latest_finish,
        'task_it_power': task_it_power,
        'n_tasks': n
    }


def get_candidate_regions(source_region_idx, max_latency, latency_matrix):
    """根据MaxLatency约束筛选候选执行区域"""
    candidates = []
    for r in range(N_REGIONS):
        if latency_matrix[source_region_idx, r] <= max_latency:
            candidates.append(r)
    return candidates


if __name__ == '__main__':
    data_dir = r'D:\数学建模\26数学建模暑期模拟训练\第二次训练题目（三选一）\C题 面向算电协同的多目标调度优化研究'
    data = load_all_data(data_dir)
    tasks = prepare_task_arrays(data['workload'])

    print(f"\n=== 数据验证 ===")
    print(f"任务总数: {tasks['n_tasks']}")
    print(f"任务类型分布: {pd.Series(tasks['task_types']).value_counts().to_dict()}")
    print(f"GPU需求总量: {tasks['gpu_demands'].sum():.0f}")
