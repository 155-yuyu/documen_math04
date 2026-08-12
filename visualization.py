"""
visualization.py - 可视化模块
功能：调度甘特图、SOC轨迹、充放电策略、对比柱状图、Pareto前沿、场景雷达图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REGIONS = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
REGION_COLORS = {
    'RegionA': '#FF6B6B', 'RegionB': '#FF8E72', 'RegionC': '#FFB347',
    'RegionD': '#4ECDC4', 'RegionE': '#95E1A3', 'RegionF': '#6BCB77'
}


def plot_6metric_comparison(baseline_m, opt_m, output_path):
    """6维指标对比柱状图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('算-储-电协同优化 6维指标对比', fontsize=16, fontweight='bold')

    labels = ['基准', '优化']
    colors = ['#95A5A6', '#E74C3C']

    # 1. 运行成本
    ax = axes[0][0]
    vals = [baseline_m['total_cost']/1e4, opt_m['total_cost']/1e4]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel('成本 (万元)', fontsize=11)
    ax.set_title('F1: 运行成本', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.0f}', ha='center', fontsize=11)

    # 2. 碳排放
    ax = axes[0][1]
    vals = [baseline_m['total_carbon'], opt_m['total_carbon']]
    bars = ax.bar(labels, vals, color=['#95A5A6', '#27AE60'], width=0.5)
    ax.set_ylabel('碳排放 (tCO2)', fontsize=11)
    ax.set_title('F2: 碳排放', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.0f}', ha='center', fontsize=11)

    # 3. 时延
    ax = axes[0][2]
    vals = [baseline_m.get('total_latency', 0), opt_m.get('total_latency', 0)]
    bars = ax.bar(labels, vals, color=['#95A5A6', '#F39C12'], width=0.5)
    ax.set_ylabel('加权时延 (ms*GPU)', fontsize=11)
    ax.set_title('F3: 网络时延', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.0f}', ha='center', fontsize=11)

    # 4. QoS
    ax = axes[1][0]
    vals = [baseline_m.get('qos_rate', 1.0)*100, opt_m.get('qos_rate', 1.0)*100]
    bars = ax.bar(labels, vals, color=['#95A5A6', '#3498DB'], width=0.5)
    ax.set_ylabel('按时完成率 (%)', fontsize=11)
    ax.set_title('F4: 服务质量', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.1f}', ha='center', fontsize=11)

    # 5. 新能源利用率
    ax = axes[1][1]
    vals = [baseline_m.get('renewable_utilization', 0)*100, opt_m.get('renewable_utilization', 0)*100]
    bars = ax.bar(labels, vals, color=['#95A5A6', '#9B59B6'], width=0.5)
    ax.set_ylabel('利用率 (%)', fontsize=11)
    ax.set_title('F5: 新能源利用率', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.1f}', ha='center', fontsize=11)

    # 6. 峰值净购电
    ax = axes[1][2]
    vals = [baseline_m.get('peak_net_import', 0), opt_m.get('peak_net_import', 0)]
    bars = ax.bar(labels, vals, color=['#95A5A6', '#E67E22'], width=0.5)
    ax.set_ylabel('峰值净购电 (MW)', fontsize=11)
    ax.set_title('F6: 峰值净购电', fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f'{v:.1f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 6维指标对比图已保存")


def plot_soc_trajectory(all_sol, data, output_path, t_start=0, t_end=480):
    """SOC轨迹图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f'优化后SOC轨迹 (第{t_start}-{t_end-1}小时)', fontsize=16, fontweight='bold')
    hours = np.arange(t_start, t_end)

    for ri, r in enumerate(REGIONS):
        ax = axes[ri//2][ri%2]
        if ri not in all_sol:
            continue
        ax.plot(hours, all_sol[ri]['soc'][t_start:t_end],
                color=REGION_COLORS[r], linewidth=1, alpha=0.8)
        ax.axhline(y=data['init_soc'][ri], color='green', linestyle=':', alpha=0.5, label='初始SOC')
        ax.axhline(y=data['min_soc'][ri], color='red', linestyle='--', alpha=0.3, label='最小SOC')
        ax.axhline(y=data['storage_cap'][ri], color='blue', linestyle='--', alpha=0.3, label='容量上限')
        ax.set_ylabel('SOC (MWh)', fontsize=10)
        ax.set_title(f'{r} (容量{int(data["storage_cap"][ri])}MWh)', fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    axes[2][0].set_xlabel('小时', fontsize=11)
    axes[2][1].set_xlabel('小时', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] SOC轨迹图已保存")


def plot_net_import(all_sol, output_path, t_start=0, t_end=480):
    """系统净购电曲线"""
    fig, ax = plt.subplots(figsize=(16, 6))
    hours = np.arange(t_start, t_end)

    total_net = np.zeros(t_end - t_start)
    for ri in range(6):
        if ri in all_sol:
            total_net += all_sol[ri]['p_net'][t_start:t_end]

    colors_pos = ['#E74C3C' if v > 0 else '#27AE60' for v in total_net]
    ax.bar(hours, total_net, color=colors_pos, alpha=0.7, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('系统净购电 (MW)', fontsize=12)
    ax.set_title(f'优化后系统净购电曲线 (第{t_start}-{t_end-1}小时)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 净购电曲线图已保存")


def plot_charge_discharge(all_sol, output_path, t_start=0, t_end=240):
    """充放电策略图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f'优化后充放电策略 (第{t_start}-{t_end-1}小时)', fontsize=16, fontweight='bold')
    hours = np.arange(t_start, t_end)

    for ri, r in enumerate(REGIONS):
        ax = axes[ri//2][ri%2]
        if ri not in all_sol:
            continue
        ax.bar(hours, all_sol[ri]['p_charge'][t_start:t_end],
               color='#3498DB', alpha=0.7, label='充电', width=1.0)
        ax.bar(hours, -all_sol[ri]['p_discharge'][t_start:t_end],
               color='#E74C3C', alpha=0.7, label='放电', width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('功率 (MW)', fontsize=10)
        ax.set_title(r, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[2][0].set_xlabel('小时')
    axes[2][1].set_xlabel('小时')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 充放电策略图已保存")


def plot_region_comparison(all_sol, data, output_path):
    """分区域对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('分区域算-储-电协同优化效果', fontsize=16, fontweight='bold')
    x = np.arange(len(REGIONS))
    width = 0.35

    # 1. 成本
    ax = axes[0][0]
    b_vals = [np.sum(data['baseline_grid_buy'][ri, :2400]*data['electricity_price'][ri, :2400] -
              data['baseline_grid_sell'][ri, :2400]*data['sell_price'][ri, :2400])/1e4
              for ri in range(6)]
    o_vals = [np.sum(all_sol[ri]['p_buy'][:2400]*data['electricity_price'][ri, :2400] -
               all_sol[ri]['p_sell'][:2400]*data['sell_price'][ri, :2400])/1e4
               if ri in all_sol else 0 for ri in range(6)]
    ax.bar(x - width/2, b_vals, width, label='基准', color='#95A5A6')
    ax.bar(x + width/2, o_vals, width, label='优化', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('成本 (万元)'); ax.set_title('分区域运行成本')
    ax.legend(); ax.grid(alpha=0.3)

    # 2. 碳排
    ax = axes[0][1]
    b_vals = [np.sum(data['baseline_grid_buy'][ri, :2400]*data['carbon_intensity'][ri, :2400]) for ri in range(6)]
    o_vals = [np.sum(all_sol[ri]['p_buy'][:2400]*data['carbon_intensity'][ri, :2400]) if ri in all_sol else 0 for ri in range(6)]
    ax.bar(x - width/2, b_vals, width, label='基准', color='#95A5A6')
    ax.bar(x + width/2, o_vals, width, label='优化', color='#27AE60')
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('碳排放 (tCO2)'); ax.set_title('分区域碳排放')
    ax.legend(); ax.grid(alpha=0.3)

    # 3. 新能源利用率
    ax = axes[1][0]
    b_vals = [np.sum(data['baseline_used_renewable'][ri, :2400]+data['baseline_renewable_charge'][ri, :2400]+data['baseline_grid_sell'][ri, :2400])/
              np.sum(data['available_renewable'][ri, :2400])*100
              if np.sum(data['available_renewable'][ri, :2400])>0 else 0
              for ri in range(6)]
    o_vals = [np.sum(all_sol[ri]['p_ren_dir'][:2400]+all_sol[ri]['p_ren_ch'][:2400]+all_sol[ri]['p_ren_sell'][:2400])/
              np.sum(data['available_renewable'][ri, :2400])*100
              if ri in all_sol and np.sum(data['available_renewable'][ri, :2400])>0 else 0 for ri in range(6)]
    ax.bar(x - width/2, b_vals, width, label='基准', color='#95A5A6')
    ax.bar(x + width/2, o_vals, width, label='优化', color='#F39C12')
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('利用率 (%)'); ax.set_title('新能源利用率')
    ax.legend(); ax.grid(alpha=0.3)

    # 4. 峰值净购电
    ax = axes[1][1]
    b_vals = [np.max(data['baseline_net_import'][ri, :2400]) for ri in range(6)]
    o_vals = [np.max(all_sol[ri]['p_net'][:2400]) if ri in all_sol else 0 for ri in range(6)]
    ax.bar(x - width/2, b_vals, width, label='基准', color='#95A5A6')
    ax.bar(x + width/2, o_vals, width, label='优化', color='#3498DB')
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('峰值净购电 (MW)'); ax.set_title('分区域峰值净购电')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 分区域对比图已保存")


def plot_pareto(pareto_points, output_path):
    """Pareto前沿图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Pareto前沿分析', fontsize=16, fontweight='bold')

    costs = [p['total_cost']/1e4 for p in pareto_points]
    carbons = [p['total_carbon'] for p in pareto_points]
    labels = [p['label'] for p in pareto_points]

    ax = axes[0]
    ax.scatter(costs, carbons, c=range(len(pareto_points)), cmap='viridis', s=120, zorder=5)
    for i, l in enumerate(labels):
        ax.annotate(l, (costs[i], carbons[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('运行成本 (万元)')
    ax.set_ylabel('碳排放 (tCO2)')
    ax.set_title('成本-碳排放权衡')
    ax.grid(alpha=0.3)

    # 时延 vs QoS
    latencies = [p.get('total_latency', 0) for p in pareto_points]
    qoss = [p.get('qos_rate', 1.0)*100 for p in pareto_points]
    ax = axes[1]
    ax.scatter(latencies, qoss, c=range(len(pareto_points)), cmap='viridis', s=120, zorder=5)
    for i, l in enumerate(labels):
        ax.annotate(l, (latencies[i], qoss[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('加权网络时延')
    ax.set_ylabel('QoS按时完成率 (%)')
    ax.set_title('时延-QoS权衡')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] Pareto前沿图已保存")


def plot_scenario_radar(scenarios, output_path, baseline_m=None):
    """场景对比雷达图"""
    metrics_keys = ['total_cost', 'total_carbon', 'total_latency',
                    'renewable_utilization', 'peak_net_import']
    metric_labels = ['成本', '碳排', '时延', '利用率', '峰值购电']

    n_metrics = len(metrics_keys)
    n_scenarios = len(scenarios)

    # 归一化 (每个指标除以最大值)
    all_vals = []
    for _, m in scenarios:
        vals = [m.get(k, 0) for k in metrics_keys]
        all_vals.append(vals)
    if baseline_m:
        all_vals.append([baseline_m.get(k, 0) for k in metrics_keys])

    max_vals = [max(v) for v in zip(*all_vals)]
    max_vals = [max(v, 1e-6) for v in max_vals]

    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#E74C3C', '#27AE60', '#3498DB', '#F39C12', '#9B59B6', '#95A5A6']

    for i, (name, m) in enumerate(scenarios):
        vals = [m.get(k, 0)/max_vals[j] for j, k in enumerate(metrics_keys)]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.15, color=colors[i % len(colors)])

    if baseline_m:
        vals = [baseline_m.get(k, 0)/max_vals[j] for j, k in enumerate(metrics_keys)]
        vals += vals[:1]
        ax.plot(angles, vals, 'o--', linewidth=2, label='基准', color='#95A5A6')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_title('场景对比雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 场景雷达图已保存")


def plot_scenario_bar(scenarios, output_path, baseline_m=None):
    """场景对比柱状图"""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('场景分析对比', fontsize=16, fontweight='bold')

    names = [s[0] for s in scenarios]
    x = np.arange(n_scenarios)
    width = 0.35

    # 1. 成本
    ax = axes[0][0]
    vals = [s[1]['total_cost']/1e4 for s in scenarios]
    ax.bar(x, vals, width, color='#E74C3C', alpha=0.7)
    if baseline_m:
        ax.axhline(y=baseline_m['total_cost']/1e4, color='gray', linestyle='--', label='基准')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_ylabel('成本 (万元)'); ax.set_title('运行成本')
    ax.legend(); ax.grid(alpha=0.3)

    # 2. 碳排
    ax = axes[0][1]
    vals = [s[1]['total_carbon'] for s in scenarios]
    ax.bar(x, vals, width, color='#27AE60', alpha=0.7)
    if baseline_m:
        ax.axhline(y=baseline_m['total_carbon'], color='gray', linestyle='--', label='基准')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_ylabel('碳排放 (tCO2)'); ax.set_title('碳排放')
    ax.legend(); ax.grid(alpha=0.3)

    # 3. 利用率
    ax = axes[1][0]
    vals = [s[1]['renewable_utilization']*100 for s in scenarios]
    ax.bar(x, vals, width, color='#F39C12', alpha=0.7)
    if baseline_m:
        ax.axhline(y=baseline_m['renewable_utilization']*100, color='gray', linestyle='--', label='基准')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_ylabel('利用率 (%)'); ax.set_title('新能源利用率')
    ax.legend(); ax.grid(alpha=0.3)

    # 4. 峰值净购电
    ax = axes[1][1]
    vals = [s[1]['peak_net_import'] for s in scenarios]
    ax.bar(x, vals, width, color='#3498DB', alpha=0.7)
    if baseline_m:
        ax.axhline(y=baseline_m['peak_net_import'], color='gray', linestyle='--', label='基准')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_ylabel('峰值净购电 (MW)'); ax.set_title('峰值净购电')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 场景对比柱状图已保存")


def plot_task_distribution(schedule_df, output_path):
    """任务迁移分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('任务调度分布', fontsize=16, fontweight='bold')

    # 1. 源区域 vs 执行区域
    ax = axes[0]
    source_counts = schedule_df['SourceRegion'].value_counts().reindex(REGIONS, fill_value=0)
    exec_counts = schedule_df['ExecRegion'].value_counts().reindex(REGIONS, fill_value=0)
    x = np.arange(len(REGIONS))
    width = 0.35
    ax.bar(x - width/2, source_counts.values, width, label='源区域', color='#95A5A6')
    ax.bar(x + width/2, exec_counts.values, width, label='执行区域', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('任务数'); ax.set_title('任务源区域 vs 执行区域')
    ax.legend(); ax.grid(alpha=0.3)

    # 2. 按任务类型
    ax = axes[1]
    types = ['RealTimeInference', 'BatchInference', 'AITraining']
    type_colors = ['#FF6B6B', '#4ECDC4', '#95E1A3']
    for i, t in enumerate(types):
        sub = schedule_df[schedule_df['TaskType'] == t]
        exec_counts = sub['ExecRegion'].value_counts().reindex(REGIONS, fill_value=0)
        ax.bar(x + (i-1)*width/2, exec_counts.values, width*0.8, label=t, color=type_colors[i])
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, rotation=30)
    ax.set_ylabel('任务数'); ax.set_title('按任务类型的执行区域分布')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 任务分布图已保存")
