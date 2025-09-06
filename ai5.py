
import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time
import os

# ===================================================================
# 1. 常量和全局参数定义
# ===================================================================
# 物理常量
g = 9.8
R_eff = 10.0
smoke_duration = 20.0
sink_speed = 3.0
dt_eval = 0.1

# 目标和导弹信息
O = np.array([0.0, 0.0, 0.0])
T_true = np.array([0.0, 200.0, 0.0])
vm = 300.0
MISSILES_INFO = {
    'M1': {'M0': np.array([20000.0, 0.0, 2000.0])},
    'M2': {'M0': np.array([19000.0, 600.0, 2100.0])},
    'M3': {'M0': np.array([18000.0, -600.0, 1900.0])}
}

# 无人机信息
DRONES_INFO = {
    'FY1': {'F0': np.array([17800.0, 0.0, 1800.0])},
    'FY2': {'F0': np.array([12000.0, 1400.0, 1400.0])},
    'FY3': {'F0': np.array([6000.0, -3000.0, 700.0])},
    'FY4': {'F0': np.array([11000.0, 2000.0, 1800.0])},
    'FY5': {'F0': np.array([13000.0, -2000.0, 1300.0])}
}

# 约束
V_UAV_BOUNDS = (70.0, 140.0)
TAU_BOUNDS = (0.2, 12.0)


# ===================================================================
# 2. 核心物理模型与计算函数
# ===================================================================

def pre_calculate_missile_trajectories(missiles):
    """预计算所有导弹的轨迹"""
    trajectories = {}
    for name, data in missiles.items():
        um = (O - data['M0']) / np.linalg.norm(O - data['M0'])
        T_hit = np.linalg.norm(O - data['M0']) / vm
        times = np.arange(0, T_hit + dt_eval, dt_eval)
        positions = data['M0'] + vm * times[:, np.newaxis] * um
        trajectories[name] = {'times': times, 'positions': positions, 'T_hit': T_hit}
    return trajectories


def inverse_model(explosion_point, F0):
    """逆向动力学模型，检查可达性"""
    x_exp, y_exp, z_exp, t_exp = explosion_point
    x_f0, y_f0, z_f0 = F0

    if z_exp > z_f0 or t_exp <= 0 or z_f0 - z_exp < 0: return {'is_reachable': False}
    tau = np.sqrt(2 * (z_f0 - z_exp) / g)
    if not (TAU_BOUNDS[0] <= tau <= TAU_BOUNDS[1]): return {'is_reachable': False}

    t_drop = t_exp - tau
    if t_drop < 0: return {'is_reachable': False}

    delta_x, delta_y = x_exp - x_f0, y_exp - y_f0
    horizontal_dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
    v = horizontal_dist / t_exp if t_exp > 1e-6 else np.inf
    if not (V_UAV_BOUNDS[0] <= v <= V_UAV_BOUNDS[1]): return {'is_reachable': False}

    theta = np.arctan2(delta_y, delta_x)
    return {'is_reachable': True, 'strategy': {'v_uav': v, 'theta': theta, 't_drop': t_drop, 'tau': tau}}


def get_shielding_mask(explosion_point, missile_trajectory):
    """返回单个起爆点对单个导弹的布尔遮蔽掩码"""
    full_time_mask = np.zeros_like(missile_trajectory['times'], dtype=bool)
    x_exp, y_exp, z_exp, t_exp = explosion_point
    t_start_smoke = t_exp
    t_end_smoke = min(t_start_smoke + smoke_duration, missile_trajectory['T_hit'])

    times = missile_trajectory['times']
    start_idx, end_idx = np.searchsorted(times, [t_start_smoke, t_end_smoke])
    if start_idx >= end_idx: return full_time_mask

    valid_times = times[start_idx:end_idx]
    Mt_positions = missile_trajectory['positions'][start_idx:end_idx]
    Cm_positions = np.array([x_exp, y_exp, z_exp]) + np.array([0, 0, 1]) * (-sink_speed * (valid_times - t_exp))[
        :, np.newaxis]

    vec_mt = T_true - Mt_positions
    len_sq = np.sum(vec_mt ** 2, axis=1)
    safe_len_sq = np.where(len_sq > 1e-9, len_sq, 1e-9)
    vec_cm = Cm_positions - Mt_positions
    t_proj = np.clip(np.sum(vec_cm * vec_mt, axis=1) / safe_len_sq, 0.0, 1.0)
    distances = np.linalg.norm(Cm_positions - (Mt_positions + t_proj[:, np.newaxis] * vec_mt), axis=1)

    full_time_mask[start_idx:end_idx] = (distances <= R_eff)
    return full_time_mask


# ===================================================================
# 3. 协同优化目标函数
# ===================================================================

def collaborative_objective_function(X, drone_names, missile_names, missile_trajectories):
    """协同优化目标函数，计算遮蔽时间的并集"""
    num_drones = len(drone_names)
    bombs_per_drone = len(X) // 4 // num_drones
    all_explosion_points = X.reshape(num_drones * bombs_per_drone, 4)
    drone_t_drop_times = [[] for _ in range(num_drones)]

    for i, point in enumerate(all_explosion_points):
        drone_idx = i // bombs_per_drone
        reach_check = inverse_model(point, DRONES_INFO[drone_names[drone_idx]]['F0'])
        if not reach_check['is_reachable']: return 1e9
        drone_t_drop_times[drone_idx].append(reach_check['strategy']['t_drop'])

    for times in drone_t_drop_times:
        times.sort()
        for i in range(1, len(times)):
            if times[i] - times[i - 1] < 1.0: return 1e9

    total_union_coverage = 0
    for m_name in missile_names:
        final_union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in all_explosion_points:
            final_union_mask = np.logical_or(final_union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        total_union_coverage += np.sum(final_union_mask) * dt_eval

    return -total_union_coverage


# ===================================================================
# 4. 主执行流程
# ===================================================================

def save_results_to_excel(results_dict, filename):
    """将最终结果保存到Excel文件"""
    data_list = []
    for drone_name, result_data in results_dict.items():
        for i, strategy in enumerate(result_data['strategies']):
            point = strategy['explosion_point']
            data_list.append({
                '无人机编号': drone_name,
                '烟幕干扰弹编号': i + 1,
                '无人机运动方向 (度)': np.degrees(strategy['theta']),
                '无人机运动速度 (m/s)': strategy['v_uav'],
                '烟幕干扰弹投放时间 (s)': strategy['t_drop'],
                '烟幕干扰弹起爆时间 (s)': strategy['t_drop'] + strategy['tau'],
                '烟幕干扰弹起爆延迟 (s)': strategy['tau'],
                '起爆点x': point[0], '起爆点y': point[1], '起爆点z': point[2],
            })
    if not data_list: return
    df = pd.DataFrame(data_list).round(4)
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"\nResults successfully saved to '{filename}'")


def solve_problem4_collaborative(missile_trajectories):
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 4 (COLLABORATIVE)")
    print("3 Drones (FY1, FY2, FY3), 1 bomb each, against Missile M1")
    print("=" * 50)

    drone_names, missile_names = ['FY1', 'FY2', 'FY3'], ['M1']
    bounds = []
    max_t_hit = missile_trajectories['M1']['T_hit']
    for name in drone_names:
        F0 = DRONES_INFO[name]['F0']
        bounds.extend([(0, F0[0] + 2000), (-3000, 3000), (0, F0[2] - 1), (1.0, max_t_hit)])

    start_time = time.time()
    result = differential_evolution(
        func=collaborative_objective_function, bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories),
        maxiter=300, popsize=30, disp=True, workers=-1, polish=True
    )
    print(f"Optimization finished in {time.time() - start_time:.2f} s.")

    if result.success and result.fun < 1e8:
        max_coverage_union = -result.fun
        print(f"\n[Problem 4 RESULT] Maximum Collaborative Coverage Time (Union): {max_coverage_union:.4f} s")

        all_points = result.x.reshape(len(drone_names), 4)
        results_dict = {}
        for i, name in enumerate(drone_names):
            point = all_points[i]
            reach_info = inverse_model(point, DRONES_INFO[name]['F0'])
            strategy = reach_info['strategy']
            strategy['explosion_point'] = point
            results_dict[name] = {'strategies': [strategy]}

        save_results_to_excel(results_dict, "result2.xlsx")
    else:
        print("\n[Problem 4 FAILED] Optimization did not converge.")


def solve_problem5(missile_trajectories):
    print("\n" + "=" * 50)
    print("              SOLVING PROBLEM 5")
    print("5 Drones, up to 3 bombs each, against Missiles M1, M2, M3")
    print("=" * 50)

    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    missile_names = ['M1', 'M2', 'M3']
    num_bombs = 3
    problem5_results = {}

    for name in drone_names:
        print(f"\n----- Optimizing for Drone: {name} -----")
        bounds = []
        max_t_hit = max(missile_trajectories[m]['T_hit'] for m in missile_names)
        F0 = DRONES_INFO[name]['F0']
        single_bomb_bounds = [(0, F0[0] + 2000), (-4000, 4000), (0, F0[2] - 1), (1.0, max_t_hit)]
        for _ in range(num_bombs): bounds.extend(single_bomb_bounds)

        result = differential_evolution(
            func=collaborative_objective_function, bounds=bounds,
            args=([name], missile_names, missile_trajectories),
            maxiter=300, popsize=30, disp=False, workers=-1, polish=True
        )

        if result.success and result.fun < 1e8:
            coverage = -result.fun
            print(f"  - SUCCESS: Max coverage contribution found: {coverage:.4f} s")
            points = result.x.reshape(num_bombs, 4)
            strategies = []
            for point in points:
                reach_info = inverse_model(point, F0)
                strategy = reach_info['strategy']
                strategy['explosion_point'] = point
                strategies.append(strategy)
            strategies.sort(key=lambda s: s['t_drop'])
            problem5_results[name] = {'strategies': strategies}
        else:
            print(f"  - FAILED: Optimization did not converge for {name}.")
            problem5_results[name] = {'strategies': []}

    save_results_to_excel(problem5_results, "result3.xlsx")

    # 最后，计算并显示问题5的总遮蔽时长
    grand_total_coverage = 0
    all_final_points = []
    for drone_name, res_data in problem5_results.items():
        for strat in res_data['strategies']:
            all_final_points.append(strat['explosion_point'])

    for m_name in missile_names:
        final_union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in all_final_points:
            final_union_mask = np.logical_or(final_union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        grand_total_coverage += np.sum(final_union_mask) * dt_eval

    print(
        f"\n[Problem 5 RESULT] Grand Total Collaborative Coverage Time (Union across all drones): {grand_total_coverage:.4f} s")


if __name__ == "__main__":
    print("Step 1: Pre-calculating all missile trajectories...")
    all_missile_trajectories = pre_calculate_missile_trajectories(MISSILES_INFO)
    print("Done.")

    # 解决问题4 (协同版)
    solve_problem4_collaborative(all_missile_trajectories)

    # 解决问题5 (内部协同，外部汇总)
    solve_problem5(all_missile_trajectories)

    print("\n\nAll tasks completed.")