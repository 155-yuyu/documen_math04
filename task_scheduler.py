"""
task_scheduler.py - 上层碳感知任务调度模块
功能：分层启发式调度 (RT固定 + Batch/Training贪心优化)
增强：储能感知评分（新能源富余时段+区域给予额外奖励）
"""

import numpy as np
from data_loader import (
    REGIONS, REGION_TO_IDX, N_REGIONS, T_TOTAL, T_MAIN,
    TASK_TYPES, TASK_TYPE_TO_IDX, GPU_POWER,
    compute_overlap, get_duration_hours, get_candidate_regions
)


class StorageAwareScheduler:
    """
    碳感知+储能感知任务调度器
    策略：
    1. RealTimeInference: 固定源区域, 到达即开工 (不可优化)
    2. BatchInference: 在时延约束候选区域内贪心优化
    3. AITraining: 全区域贪心优化 (时延约束=150ms, 所有区域可达)

    储能感知增强：评分中引入新能源富余因子，当目标区域当前时段
    新能源富余且储能可充电时，给予额外奖励，促进"算随绿走"。
    """

    def __init__(self, data, tasks, weights=None, search_window=72,
                 carbon_constraint=None, renewable_factor=1.0):
        """
        data: load_all_data()返回的字典
        tasks: prepare_task_arrays()返回的字典
        weights: (w_cost, w_carbon, w_latency, w_renewable) 权重
        search_window: 开工时段搜索窗口（小时）
        carbon_constraint: 碳约束上限 (tCO2)，None表示无约束
        renewable_factor: 新能源出力系数 (1.0=基准, 0.8=低, 1.2=高)
        """
        self.data = data
        self.tasks = tasks
        self.weights = weights if weights is not None else (0.25, 0.25, 0.25, 0.25)
        self.search_window = search_window
        self.carbon_constraint = carbon_constraint
        self.renewable_factor = renewable_factor

        # 调整新能源出力
        self._avail_ren = data['available_renewable'] * renewable_factor

        # 状态数组 [6, 2407]
        self.gpu_usage = np.zeros((N_REGIONS, T_TOTAL))
        self.ai_it_load = np.zeros((N_REGIONS, T_TOTAL))

        # 调度结果
        self.schedule = {
            'exec_region': np.full(tasks['n_tasks'], -1, dtype=np.int32),
            'start_hour': np.full(tasks['n_tasks'], -1, dtype=np.int32),
        }

        # 归一化参考值
        self._compute_norms()

    def _compute_norms(self):
        """计算归一化参考值"""
        d = self.data
        self.norm_cost = np.mean(d['cost_rate']) * np.mean(self.tasks['task_it_power']) * 4.0
        self.norm_carbon = np.mean(d['carbon_rate']) * np.mean(self.tasks['task_it_power']) * 4.0
        self.norm_latency = 82.0  # ms (最大跨区时延)
        self.norm_renewable = np.mean(self._avail_ren) * 4.0

    def _assign_task(self, task_idx, region_idx, start_hour, overlap, num_hours):
        """将任务分配到指定区域和时段，更新状态"""
        gpu_demand = self.tasks['gpu_demands'][task_idx]
        it_power = self.tasks['task_it_power'][task_idx]
        end_hour = min(start_hour + num_hours, T_TOTAL)

        self.gpu_usage[region_idx, start_hour:end_hour] += gpu_demand * overlap[:end_hour - start_hour]
        self.ai_it_load[region_idx, start_hour:end_hour] += it_power * overlap[:end_hour - start_hour]

        self.schedule['exec_region'][task_idx] = region_idx
        self.schedule['start_hour'][task_idx] = start_hour

    def _evaluate_placement(self, task_idx, region_idx, overlap, num_hours):
        """
        评估任务在指定区域的所有候选时段
        返回: (best_start, best_score, feasible_mask, scores)
        """
        d = self.data
        t = self.tasks
        arrival = t['arrival_hours'][task_idx]
        latest = t['latest_finish'][task_idx]
        gpu_demand = t['gpu_demands'][task_idx]
        it_power = t['task_it_power'][task_idx]
        source_r = t['source_region_idx'][task_idx]

        s_min = int(arrival)
        s_max = int(min(latest - num_hours, arrival + self.search_window, T_TOTAL - num_hours - 1))
        if s_max < s_min:
            return None, np.inf, None, None

        s_range = np.arange(s_min, s_max + 1)
        n_starts = len(s_range)

        t_indices = s_range[:, None] + np.arange(num_hours)[None, :]

        # ========== 容量约束检查 ==========
        gpu_feasible = np.all(
            self.gpu_usage[region_idx, t_indices] + gpu_demand * overlap[None, :] <= d['available_gpu'][region_idx],
            axis=1)
        it_feasible = np.all(
            self.ai_it_load[region_idx, t_indices] + it_power * overlap[None, :] <= d['max_it_power'][region_idx],
            axis=1)
        total_it = d['nonai_it_load'][region_idx, t_indices] + self.ai_it_load[region_idx, t_indices] + it_power * overlap[None, :]
        facility_feasible = np.all(
            total_it * d['pue'][region_idx] <= d['max_facility_power'][region_idx],
            axis=1)

        feasible = gpu_feasible & it_feasible & facility_feasible
        if not np.any(feasible):
            return None, np.inf, feasible, None

        # ========== 目标函数计算 ==========
        w1, w2, w3, w4 = self.weights

        # 1. 成本
        cost_vals = np.dot(d['cost_rate'][region_idx, t_indices], overlap) * it_power

        # 2. 碳排
        carbon_vals = np.dot(d['carbon_rate'][region_idx, t_indices], overlap) * it_power

        # 3. 时延 (固定值)
        latency_val = d['latency_matrix'][source_r, region_idx]

        # 4. 新能源利用率 (储能感知增强)
        task_facility_power = it_power * d['pue'][region_idx]
        renewable_avail = np.maximum(0, self._avail_ren[region_idx, t_indices] -
                                     (d['nonai_it_load'][region_idx, t_indices] + self.ai_it_load[region_idx, t_indices]) * d['pue'][region_idx])
        renewable_used = np.minimum(task_facility_power * overlap[None, :], renewable_avail)
        renewable_vals = np.sum(renewable_used, axis=1)

        # 加权评分 (越小越好, renewable越大越好所以取负)
        scores = (w1 * cost_vals / self.norm_cost +
                  w2 * carbon_vals / self.norm_carbon +
                  w3 * np.full(n_starts, latency_val / self.norm_latency) -
                  w4 * renewable_vals / self.norm_renewable)

        scores[~feasible] = np.inf

        best_idx = np.argmin(scores)
        if scores[best_idx] == np.inf:
            return None, np.inf, feasible, scores

        return int(s_range[best_idx]), float(scores[best_idx]), feasible, scores

    def schedule_fixed_tasks(self):
        """调度RealTimeInference任务：固定源区域，到达即开工"""
        t = self.tasks
        rt_mask = (t['task_types'] == 'RealTimeInference')
        rt_indices = np.where(rt_mask)[0]

        n_assigned = 0
        n_failed = 0
        for idx in rt_indices:
            region_idx = int(t['source_region_idx'][idx])
            start_hour = int(t['arrival_hours'][idx])
            duration_min = int(t['durations_min'][idx])
            overlap, num_hours = compute_overlap(duration_min)
            end_hour = min(start_hour + num_hours, T_TOTAL)

            gpu_ok = np.all(self.gpu_usage[region_idx, start_hour:end_hour] +
                            t['gpu_demands'][idx] * overlap[:end_hour - start_hour] <=
                            self.data['available_gpu'][region_idx])
            it_ok = np.all(self.ai_it_load[region_idx, start_hour:end_hour] +
                           t['task_it_power'][idx] * overlap[:end_hour - start_hour] <=
                           self.data['max_it_power'][region_idx])
            total_it = self.data['nonai_it_load'][region_idx, start_hour:end_hour] + \
                       self.ai_it_load[region_idx, start_hour:end_hour] + \
                       t['task_it_power'][idx] * overlap[:end_hour - start_hour]
            fac_ok = np.all(total_it * self.data['pue'][region_idx] <= self.data['max_facility_power'][region_idx])

            if gpu_ok and it_ok and fac_ok:
                self._assign_task(idx, region_idx, start_hour, overlap, num_hours)
                n_assigned += 1
            else:
                self._assign_task(idx, region_idx, start_hour, overlap, num_hours)
                n_failed += 1

        print(f"[RT调度] 分配 {n_assigned}/{len(rt_indices)} 个实时推理任务 (约束违反: {n_failed})")
        return n_assigned, n_failed

    def schedule_flexible_tasks(self):
        """调度BatchInference和AITraining任务：贪心优化"""
        t = self.tasks
        flex_mask = (t['task_types'] == 'BatchInference') | (t['task_types'] == 'AITraining')
        flex_indices = np.where(flex_mask)[0]

        flex_indices = flex_indices[np.argsort(
            t['arrival_hours'][flex_indices] * 1000 - t['gpu_demands'][flex_indices]
        )]

        n_assigned = 0
        n_failed = 0
        for idx in flex_indices:
            duration_min = int(t['durations_min'][idx])
            overlap, num_hours = compute_overlap(duration_min)
            source_r = int(t['source_region_idx'][idx])
            max_lat = int(t['max_latencies'][idx])

            candidates = get_candidate_regions(source_r, max_lat, self.data['latency_matrix'])

            best_region = -1
            best_start = -1
            best_score = np.inf

            for r_idx in candidates:
                start, score, _, _ = self._evaluate_placement(idx, r_idx, overlap, num_hours)
                if start is not None and score < best_score:
                    best_score = score
                    best_start = start
                    best_region = r_idx

            if best_region >= 0:
                self._assign_task(idx, best_region, best_start, overlap, num_hours)
                n_assigned += 1
            else:
                region_idx = source_r
                start_hour = int(t['arrival_hours'][idx])
                end_hour = min(start_hour + num_hours, T_TOTAL)
                self._assign_task(idx, region_idx, start_hour, overlap, min(num_hours, end_hour - start_hour))
                n_failed += 1

        print(f"[弹性调度] 分配 {n_assigned}/{len(flex_indices)} 个弹性任务 (回退: {n_failed})")
        return n_assigned, n_failed

    def schedule_baseline(self):
        """基准调度：所有任务在源区域+到达时间执行（无迁移）"""
        t = self.tasks
        for idx in range(t['n_tasks']):
            region_idx = int(t['source_region_idx'][idx])
            start_hour = int(t['arrival_hours'][idx])
            duration_min = int(t['durations_min'][idx])
            overlap, num_hours = compute_overlap(duration_min)
            self._assign_task(idx, region_idx, start_hour, overlap, num_hours)
        print(f"[基准调度] {t['n_tasks']} 个任务全部按源区域+到达时间分配")

    def get_ai_it_load(self):
        """返回当前AI IT负荷 [6, 2407]"""
        return self.ai_it_load.copy()

    def get_facility_load(self):
        """返回当前设施负荷 [6, 2407]"""
        it_load = self.data['nonai_it_load'] + self.ai_it_load
        return it_load * self.data['pue'][:, None]

    def get_schedule_df(self):
        """返回调度结果DataFrame"""
        import pandas as pd
        t = self.tasks
        exec_regions = [REGIONS[i] if i >= 0 else 'N/A' for i in self.schedule['exec_region']]
        source_regions = [REGIONS[i] for i in t['source_region_idx']]

        df = pd.DataFrame({
            'TaskID': t['task_ids'],
            'TaskType': t['task_types'],
            'SourceRegion': source_regions,
            'ExecRegion': exec_regions,
            'ArrivalHour': t['arrival_hours'],
            'StartHour': self.schedule['start_hour'],
            'Duration_min': t['durations_min'],
            'Duration_hours': t['duration_hours'],
            'GPU_Demand': t['gpu_demands'],
            'Task_IT_Power_MW': t['task_it_power'],
            'MaxLatency_ms': t['max_latencies'],
            'LatestFinishHour': t['latest_finish'],
            'Migrated': np.array([e != s for e, s in zip(exec_regions, source_regions)])
        })
        df['FinishHour'] = df['StartHour'] + df['Duration_hours']
        df['OnTime'] = df['FinishHour'] <= df['LatestFinishHour']
        return df

    def run(self, mode='optimized'):
        """运行调度"""
        if mode == 'baseline':
            self.schedule_baseline()
        else:
            self.schedule_fixed_tasks()
            self.schedule_flexible_tasks()
        return self.get_schedule_df()
