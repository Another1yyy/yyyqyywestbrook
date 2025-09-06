import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time
import os

# ===================================================================
# 1. 常量和全局参数定义
# ===================================================================
g, R_eff, smoke_duration, sink_speed, dt_eval = 9.8, 10.0, 20.0, 3.0, 0.1
O, T_true, vm = np.array([0., 0., 0.]), np.array([0., 200., 0.]), 300.0
MISSILES_INFO = {
    'M1': {'M0': np.array([20000., 0., 2000.])},
    'M2': {'M0': np.array([19000., 600., 2100.])},
    'M3': {'M0': np.array([18000., -600., 1900.])}
}
DRONES_INFO = {
    'FY1': {'F0': np.array([17800., 0., 1800.])},
    'FY2': {'F0': np.array([12000., 1400., 1400.])},
    'FY3': {'F0': np.array([6000., -3000., 700.])},
    'FY4': {'F0': np.array([11000., 2000., 1800.])},
    'FY5': {'F0': np.array([13000., -2000., 1300.])}
}
V_UAV_BOUNDS, TAU_BOUNDS = (70.0, 140.0), (0.2, 12.0)


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


def forward_model(strategy, F0):
    """正向模型：根据策略计算起爆点"""
    v, theta, t_drop, tau = strategy['v_uav'], strategy['theta'], strategy['t_drop'], strategy['tau']
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = F0 + v * uf * t_drop
    explode_point = drop_point + v * uf * tau + np.array([0, 0, -0.5 * g * tau ** 2])
    t_exp = t_drop + tau
    return np.array([explode_point[0], explode_point[1], explode_point[2], t_exp])


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
    t_start_smoke, t_end_smoke = t_exp, min(t_exp + smoke_duration, missile_trajectory['T_hit'])
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


def collaborative_objective_function(X, drone_names, missile_names, missile_trajectories, bombs_per_drone):
    """协同优化目标函数，计算遮蔽时间的交集或并集"""
    num_drones = len(drone_names)
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

    missile_union_masks = {}
    for m_name in missile_names:
        union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in all_explosion_points:
            union_mask = np.logical_or(union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        missile_union_masks[m_name] = union_mask

    if len(missile_names) > 1:  # 问题5逻辑：计算交集
        final_mask = np.ones_like(missile_union_masks[missile_names[0]], dtype=bool)
        for m_name in missile_names:
            final_mask = np.logical_and(final_mask, missile_union_masks[m_name])
    else:  # 问题1,2,4逻辑：计算并集 (因为只有一个导弹，并集就是它自己)
        final_mask = missile_union_masks[missile_names[0]]

    total_coverage = np.sum(final_mask) * dt_eval
    return -total_coverage


# ===================================================================
# 4. 各问题求解器
# ===================================================================

def solve_problem1(missile_trajectories):
    print("\n" + "=" * 50)
    print("              SOLVING PROBLEM 1")
    print("FY1, 1 bomb, against M1 with a FIXED strategy")
    print("=" * 50)

    F0_fy1 = DRONES_INFO['FY1']['F0']
    direction_vec = O - F0_fy1

    # 定义给定策略
    fixed_strategy = {
        'v_uav': 120.0,
        'theta': np.arctan2(direction_vec[1], direction_vec[0]),
        't_drop': 1.5,
        'tau': 3.6
    }

    # 计算起爆点
    explosion_point = forward_model(fixed_strategy, F0_fy1)

    # 计算遮蔽掩码并得出时长
    mask = get_shielding_mask(explosion_point, missile_trajectories['M1'])
    coverage_time = np.sum(mask) * dt_eval

    print(f"[Problem 1 RESULT] Effective Coverage Time: {coverage_time:.4f} s")
    print("  - Strategy Details:")
    print(f"    - Velocity: {fixed_strategy['v_uav']:.2f} m/s")
    print(f"    - Angle: {np.degrees(fixed_strategy['theta']):.2f} degrees")
    print(f"    - Drop Time: {fixed_strategy['t_drop']:.2f} s")
    print(f"    - Detonation Delay: {fixed_strategy['tau']:.2f} s")


def solve_problem2(missile_trajectories):
    print("\n" + "=" * 50)
    print("              SOLVING PROBLEM 2")
    print("Optimizing strategy for FY1, 1 bomb, against M1")
    print("=" * 50)

    drone_names, missile_names, bombs_per_drone = ['FY1'], ['M1'], 1
    bounds = []
    max_t_hit = missile_trajectories['M1']['T_hit']
    F0 = DRONES_INFO['FY1']['F0']
    bounds.extend([(0, F0[0] + 2000), (-3000, 3000), (0, F0[2] - 1), (1.0, max_t_hit)])

    result = differential_evolution(
        func=collaborative_objective_function, bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        maxiter=200, popsize=25, disp=True, workers=-1, polish=True
    )

    if result.success and result.fun < 1e8:
        max_coverage = -result.fun
        point = result.x
        reach_info = inverse_model(point, F0)
        strategy = reach_info['strategy']

        print(f"\n[Problem 2 RESULT] Maximum Coverage Time: {max_coverage:.4f} s")
        print("  - Optimal Strategy:")
        print(f"    - Velocity: {strategy['v_uav']:.2f} m/s")
        print(f"    - Angle: {np.degrees(strategy['theta']):.2f} degrees")
        print(f"    - Drop Time: {strategy['t_drop']:.2f} s")
        print(f"    - Detonation Delay: {strategy['tau']:.2f} s")
    else:
        print("\n[Problem 2 FAILED] Optimization did not converge.")


def solve_problem4_collaborative(missile_trajectories):
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 4 (COLLABORATIVE)")
    print("3 Drones (FY1, FY2, FY3), 1 bomb each, against Missile M1")
    print("=" * 50)
    drone_names, missile_names, bombs_per_drone = ['FY1', 'FY2', 'FY3'], ['M1'], 1
    bounds, max_t_hit = [], missile_trajectories['M1']['T_hit']
    for name in drone_names:
        F0 = DRONES_INFO[name]['F0']
        bounds.extend([(0, F0[0] + 2000), (-3000, 3000), (0, F0[2] - 1), (1.0, max_t_hit)])

    result = differential_evolution(
        func=collaborative_objective_function, bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        maxiter=300, popsize=30, disp=True, workers=-1, polish=True
    )

    if result.success and result.fun < 1e8:
        max_coverage_union = -result.fun
        print(f"\n[Problem 4 RESULT] Maximum Collaborative Coverage Time (Union): {max_coverage_union:.4f} s")
        all_points = result.x.reshape(len(drone_names), 4)
        results_dict = {}
        for i, name in enumerate(drone_names):
            reach_info = inverse_model(all_points[i], DRONES_INFO[name]['F0'])
            strategy = reach_info['strategy']
            strategy['explosion_point'] = all_points[i]
            results_dict[name] = {'strategies': [strategy]}
        save_results_to_excel(results_dict, "result2.xlsx")
    else:
        print("\n[Problem 4 FAILED] Optimization did not converge.")


def solve_problem5_collaborative(missile_trajectories):
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 5 (FULL COLLABORATIVE)")
    print("5 Drones, 3 bombs each, for SIMULTANEOUS shielding of M1, M2, M3")
    print("=" * 50)
    drone_names, missile_names, bombs_per_drone = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5'], ['M1', 'M2', 'M3'], 3
    bounds, max_t_hit = [], max(missile_trajectories[m]['T_hit'] for m in missile_names)
    for name in drone_names:
        F0 = DRONES_INFO[name]['F0']
        for _ in range(bombs_per_drone):
            bounds.extend([(0, 20000), (-5000, 5000), (0, F0[2] - 1), (1.0, max_t_hit)])

    result = differential_evolution(
        func=collaborative_objective_function, bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        maxiter=500, popsize=40, disp=True, workers=-1, polish=True
    )

    if result.success and result.fun < 1e8:
        max_simultaneous_coverage = -result.fun
        print(
            f"\n[Problem 5 RESULT] Maximum SIMULTANEOUS Coverage Time (Intersection): {max_simultaneous_coverage:.4f} s")
        all_points = result.x.reshape(len(drone_names) * bombs_per_drone, 4)
        results_dict = {name: {'strategies': []} for name in drone_names}
        for i, point in enumerate(all_points):
            drone_idx = i // bombs_per_drone
            drone_name = drone_names[drone_idx]
            reach_info = inverse_model(point, DRONES_INFO[drone_name]['F0'])
            strategy = reach_info['strategy']
            strategy['explosion_point'] = point
            results_dict[drone_name]['strategies'].append(strategy)
        save_results_to_excel(results_dict, "result3.xlsx")
    else:
        print("\n[Problem 5 FAILED] Optimization did not converge.")


def save_results_to_excel(results_dict, filename):
    """辅助函数，保存结果到Excel"""
    data_list = []
    for drone_name, result_data in results_dict.items():
        strategies = sorted(result_data.get('strategies', []), key=lambda s: s['t_drop'])
        for i, strategy in enumerate(strategies):
            point = strategy['explosion_point']
            data_list.append({
                '无人机编号': drone_name, '烟幕干扰弹编号': i + 1,
                '无人机运动方向 (度)': np.degrees(strategy['theta']),
                '无人机运动速度 (m/s)': strategy['v_uav'],
                '烟幕干扰弹投放时间 (s)': strategy['t_drop'],
                '烟幕干扰弹起爆时间 (s)': strategy['t_drop'] + strategy['tau'],
                '烟幕干扰弹起爆延迟 (s)': strategy['tau'],
                '起爆点x': point[0], '起爆点y': point[1], '起爆点z': point[2],
            })
    if not data_list: return
    pd.DataFrame(data_list).round(4).to_excel(filename, index=False, engine='openpyxl')
    print(f"\nResults successfully saved to '{filename}'")


# ===================================================================
# 5. 主程序入口
# ===================================================================

if __name__ == "__main__":
    print("Step 1: Pre-calculating all missile trajectories...")
    all_missile_trajectories = pre_calculate_missile_trajectories(MISSILES_INFO)
    print("Done.")

    # 依次解决各个问题
    solve_problem1(all_missile_trajectories)
    solve_problem2(all_missile_trajectories)
    solve_problem4_collaborative(all_missile_trajectories)
    solve_problem5_collaborative(all_missile_trajectories)

    print("\n\nAll tasks completed.")