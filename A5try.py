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
dt_eval = 0.05

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
def forward_model(strategy, F0):
    """
    【正向模型】根据策略计算起爆点。这个函数之前在问题1中用过。
    """
    v, theta, t_drop, tau = strategy['v_uav'], strategy['theta'], strategy['t_drop'], strategy['tau']
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = F0 + v * uf * t_drop
    explode_point = drop_point + v * uf * tau + np.array([0, 0, -0.5 * g * tau**2])
    t_exp = t_drop + tau
    return np.array([explode_point[0], explode_point[1], explode_point[2], t_exp])

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


def objective_function_intersection(X, drone_names, missile_names, missile_trajectories, bombs_per_drone):
    """
    【为问题5设计】目标函数，只计算遮蔽时间的交集。
    """
    num_drones = len(drone_names)
    all_explosion_points = X.reshape(num_drones * bombs_per_drone, 4)
    # ... (可达性与约束检查，和之前一样) ...
    drone_t_drop_times = [[] for _ in range(num_drones)]
    for i, point in enumerate(all_explosion_points):
        drone_idx = i // bombs_per_drone
        reach_check = inverse_model(point, DRONES_INFO[drone_names[drone_idx]]['F0'])
        if not reach_check['is_reachable']:

            return 1e9
        drone_t_drop_times[drone_idx].append(reach_check['strategy']['t_drop'])
    for times in drone_t_drop_times:
        times.sort()
        for i in range(1, len(times)):
            if times[i] - times[i - 1] < 1.0:
                print("这里发生了time之间差距小于1的情况")
                return 1e9

    # 1. 为每个导弹计算总的遮蔽并集掩码
    missile_union_masks = {}
    for m_name in missile_names:
        union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in all_explosion_points:
            union_mask = np.logical_or(union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        missile_union_masks[m_name] = union_mask

    # 2. 计算所有导弹掩码的交集
    final_intersection_mask = np.ones_like(missile_union_masks[missile_names[0]], dtype=bool)
    for m_name in missile_names:
        final_intersection_mask = np.logical_and(final_intersection_mask, missile_union_masks[m_name])

    total_intersection_coverage = np.sum(final_intersection_mask) * dt_eval
    return -total_intersection_coverage


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
        maxiter=500, popsize=200, disp=True, workers=-1, polish=True
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


# ===================================================================
# 4. 各问题求解器 (重构 solve_problem5)
# ===================================================================

# (solve_problem1, solve_problem2, solve_problem4, save_results_to_excel 和
# ===================================================================
# 3. 目标函数 (带软惩罚逻辑)
# ===================================================================

def objective_function_intersection_soft_penalty(X, drone_names, missile_names, missile_trajectories, bombs_per_drone):
    """
    【带软惩罚的最终版】为问题5设计的目标函数。
    如果方案无效，则返回一个与“无效程度”成正比的惩罚值。
    """
    num_drones = len(drone_names)
    all_explosion_points = X.reshape(num_drones * bombs_per_drone, 4)
    drone_t_drop_times = [[] for _ in range(num_drones)]

    # --- 阶段一：检查可达性并计算惩罚 ---

    unreachable_points_count = 0
    valid_explosion_points = []  # 只存储可达的点

    for i, point in enumerate(all_explosion_points):
        drone_idx = i // bombs_per_drone
        reach_check = inverse_model(point, DRONES_INFO[drone_names[drone_idx]]['F0'])

        if reach_check['is_reachable']:
            drone_t_drop_times[drone_idx].append(reach_check['strategy']['t_drop'])
            valid_explosion_points.append(point)
        else:
            unreachable_points_count += 1

    # --- 阶段二：检查投放间隔约束 ---

    interval_violation_count = 0
    for times in drone_t_drop_times:
        times.sort()
        for i in range(1, len(times)):
            if times[i] - times[i - 1] < 1.0:
                interval_violation_count += 1

    # --- 阶段三：根据检查结果返回适应度或惩罚值 ---

    # 如果存在任何不可达或违反间隔的点，则进入惩罚模式
    if unreachable_points_count > 0 or interval_violation_count > 0:
        # 基础惩罚值，确保任何无效解都比有效解差 (有效解返回值是负数)
        base_penalty = 1000.0

        # 惩罚与无效点的数量成正比
        # 每个不可达的点，罚100分
        # 每个违反间隔的约束，罚200分 (可以设置更高权重)
        penalty = base_penalty + (unreachable_points_count * 100) + (interval_violation_count * 200)
        return penalty

    # --- 如果所有检查都通过 (方案完全有效) ---

    # 计算交集时长
    missile_union_masks = {}
    for m_name in missile_names:
        union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in valid_explosion_points:  # 使用已验证的可达点
            union_mask = np.logical_or(union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        missile_union_masks[m_name] = union_mask

    final_intersection_mask = np.ones_like(missile_union_masks[missile_names[0]], dtype=bool)
    for m_name in missile_names:
        final_intersection_mask = np.logical_and(final_intersection_mask, missile_union_masks[m_name])

    total_intersection_coverage = np.sum(final_intersection_mask) * dt_eval

    # 返回负的遮蔽时长
    return -total_intersection_coverage




def solve_problem5(missile_trajectories):
    """
    【最终版】解决问题5，采用全局协同优化策略。
    此函数将构建一个60维的优化问题，并调用 objective_function_intersection 求解。
    """
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 5 (FULL COLLABORATIVE - INTERSECTION)")
    print("5 Drones, 3 bombs each, for SIMULTANEOUS shielding of M1, M2, M3")
    print("WARNING: This is a 60-dimensional optimization and will take significant time.")
    print("=" * 50)

    # --- 阶段一：准备 (Preparation) ---

    # 1. 定义问题的全局结构
    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    missile_names = ['M1', 'M2', 'M3']
    bombs_per_drone = 3

    # 2. 构建60维的搜索边界 (bounds)
    print("  - Building 60-dimensional search space...")
    bounds = []
    max_t_hit = max(missile_trajectories[m]['T_hit'] for m in missile_names)

    # 遍历每一架无人机，为它的3枚炸弹设定搜索范围
    for name in drone_names:
        F0 = DRONES_INFO[name]['F0']
        for _ in range(bombs_per_drone):
            # 定义单枚炸弹的[x, y, z, t]搜索范围
            single_bomb_bounds = [
                (0, 20000),  # x-range
                (-5000, 5000),  # y-range
                (0, F0[2] - 1),  # z-range
                (1.0, max_t_hit)  # t-range
            ]
            bounds.extend(single_bomb_bounds)
    print("  - Search space built successfully.")

    # --- 阶段二：执行 (Execution) ---

    print("  - Starting differential evolution...")
    start_time = time.time()

    # 调用差分进化算法，传入完整的无人机列表和60维边界
    result = differential_evolution(
        func=objective_function_intersection_soft_penalty,  # <-- 使用你已有的交集目标函数
        bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        maxiter=1000,  # 迭代次数
        popsize=1000,  # 种群大小 (popsize * 60维 > 500 是一个不错的经验法则)
        disp=True,  # 实时显示优化进程
        workers=-1,  # 使用所有CPU核心
        polish=True  # 对最终解进行精加工
    )

    print(f"  - Optimization finished in {time.time() - start_time:.2f} s.")

    # --- 阶段三：收尾 (Finalization) ---

    if result.success and result.fun < -1e-4:
        max_simultaneous_coverage = -result.fun

        # 打印最重要的结果：最终的同步遮蔽总时长
        print(
            f"\n[Problem 5 RESULT] Maximum SIMULTANEOUS Coverage Time (Intersection): {max_simultaneous_coverage:.4f} s")

        # 将60维的最优解向量 result.x 重新组织
        all_points = result.x.reshape(len(drone_names) * bombs_per_drone, 4)

        results_dict = {name: {'strategies': []} for name in drone_names}

        # 遍历所有15个最优起爆点，反向求解具体策略
        for i, point in enumerate(all_points):
            drone_idx = i // bombs_per_drone
            drone_name = drone_names[drone_idx]

            reach_info = inverse_model(point, DRONES_INFO[drone_name]['F0'])
            if reach_info['is_reachable']:  # 双重检查确保解是有效的
                # 保存完整的协同策略到Excel
                save_results_to_excel(results_dict, "result3.xlsx")
            else:
                print("\n[Problem 5 FAILED] Optimization did not find a valid non-zero solution.")
                if hasattr(result, 'message'):
                    print(f"  - Reason: {result.message}")


def objective_function_forward(X, drone_names, missile_names, missile_trajectories, bombs_per_drone):
    """
    【为问题5正向思路设计】目标函数，接收策略向量，计算交集时长。
    """
    all_explosion_points = []

    # --- 1. 解析策略向量并检查内部约束 ---
    num_drones = len(drone_names)
    params_per_drone = 2 + 2 * bombs_per_drone  # v, theta, 3*t_drop, 3*tau

    for i in range(num_drones):
        drone_name = drone_names[i]
        F0 = DRONES_INFO[drone_name]['F0']

        # 提取该无人机的所有参数
        params_start_idx = i * params_per_drone
        v, theta = X[params_start_idx], X[params_start_idx + 1]

        t_drops = []
        taus = []
        for j in range(bombs_per_drone):
            t_drops.append(X[params_start_idx + 2 + 2 * j])
            taus.append(X[params_start_idx + 3 + 2 * j])

        # 检查投放时间间隔约束
        sorted_t_drops = sorted(t_drops)
        if any(sorted_t_drops[k] < sorted_t_drops[k - 1] + 1.0 for k in range(1, bombs_per_drone)):
            return 1e9  # 违反约束，给予巨大惩罚

        # 使用正向模型计算100%可达的起爆点
        for t_drop, tau in zip(t_drops, taus):
            strategy = {'v_uav': v, 'theta': theta, 't_drop': t_drop, 'tau': tau}
            point = forward_model(strategy, F0)
            all_explosion_points.append(point)

    # --- 2. 计算同步遮蔽时长 (交集) ---
    missile_union_masks = {}
    for m_name in missile_names:
        union_mask = np.zeros_like(missile_trajectories[m_name]['times'], dtype=bool)
        for point in all_explosion_points:
            union_mask = np.logical_or(union_mask, get_shielding_mask(point, missile_trajectories[m_name]))
        missile_union_masks[m_name] = union_mask

    final_intersection_mask = np.ones_like(missile_union_masks[missile_names[0]], dtype=bool)
    for m_name in missile_names:
        final_intersection_mask = np.logical_and(final_intersection_mask, missile_union_masks[m_name])

    total_intersection_coverage = np.sum(final_intersection_mask) * dt_eval
    return -total_intersection_coverage


# ===================================================================
# 4. 基于正向思路的问题5求解器
# ===================================================================

def solve_problem5_forward(missile_trajectories):
    """
    【最终版】解决问题5，采用正向优化思路。
    """
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 5 (FORWARD OPTIMIZATION)")
    print("5 Drones, 3 bombs each, for SIMULTANEOUS shielding of M1, M2, M3")
    print("=" * 50)

    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    missile_names = ['M1', 'M2', 'M3']
    bombs_per_drone = 3

    # --- 1. 构建40维的策略边界 (bounds) ---
    print("  - Building 40-dimensional strategy search space...")
    bounds = []
    max_t_hit = max(missile_trajectories[m]['T_hit'] for m in missile_names)

    # 遍历每一架无人机，为它的策略参数设定范围
    for name in drone_names:
        # 每架无人机有 1个v, 1个theta, 3个t_drop, 3个tau
        drone_bounds = [
            V_UAV_BOUNDS,  # v_uav
            (0, 2 * np.pi),  # theta
            (0, max_t_hit - TAU_BOUNDS[1]), TAU_BOUNDS,  # t_drop1, tau1
            (0, max_t_hit - TAU_BOUNDS[1]), TAU_BOUNDS,  # t_drop2, tau2
            (0, max_t_hit - TAU_BOUNDS[1]), TAU_BOUNDS,  # t_drop3, tau3
        ]
        bounds.extend(drone_bounds)
    print("  - Search space built successfully.")

    # --- 2. 执行差分进化算法 ---
    print("  - Starting differential evolution...")
    start_time = time.time()

    result = differential_evolution(
        func=objective_function_forward,
        bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        maxiter=1000,  # 维度降低，可以适当增加迭代
        popsize=60,
        disp=True,
        workers=-1,
        polish=True
    )
    print(f"  - Optimization finished in {time.time() - start_time:.2f} s.")

    # --- 3. 处理并保存结果 ---
    if result.success and result.fun < -1e-4:
        max_simultaneous_coverage = -result.fun
        print(
            f"\n[Problem 5 RESULT] Maximum SIMULTANEOUS Coverage Time (Intersection): {max_simultaneous_coverage:.4f} s")

        # 解析40维的最优策略向量 result.x
        optimal_strategies = result.x
        results_dict = {name: {'strategies': []} for name in drone_names}
        params_per_drone = 2 + 2 * bombs_per_drone

        for i, name in enumerate(drone_names):
            params_start_idx = i * params_per_drone
            v, theta = optimal_strategies[params_start_idx], optimal_strategies[params_start_idx + 1]

            drone_strats = []
            for j in range(bombs_per_drone):
                t_drop = optimal_strategies[params_start_idx + 2 + 2 * j]
                tau = optimal_strategies[params_start_idx + 3 + 2 * j]
                strategy = {'v_uav': v, 'theta': theta, 't_drop': t_drop, 'tau': tau}

                # 计算对应的起爆点以保存到Excel
                point = forward_model(strategy, DRONES_INFO[name]['F0'])
                strategy['explosion_point'] = point
                drone_strats.append(strategy)

            # 按投放时间排序后存入
            drone_strats.sort(key=lambda s: s['t_drop'])
            results_dict[name]['strategies'] = drone_strats

        save_results_to_excel(results_dict, "result3_forward.xlsx")
    else:
        print("\n[Problem 5 FAILED] Optimization did not find a valid non-zero solution.")
        if hasattr(result, 'message'):
            print(f"  - Reason: {result.message}")


if __name__ == "__main__":
    print("Step 1: Pre-calculating all missile trajectories...")
    all_missile_trajectories = pre_calculate_missile_trajectories(MISSILES_INFO)
    print("Done.")

    # # 解决问题4 (协同版)
    # solve_problem4_collaborative(all_missile_trajectories)

    # 解决问题5 (内部协同，外部汇总)
    # solve_problem5(all_missile_trajectories)
    solve_problem5_forward(all_missile_trajectories)

    print("\n\nAll tasks completed.")