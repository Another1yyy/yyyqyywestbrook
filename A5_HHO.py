import numpy as np
import pandas as pd
import time
import os
from scipy.special import gamma # <-- 新增这一行
from A5try import (objective_function_intersection,objective_function_intersection_soft_penalty,
                   save_results_to_excel,pre_calculate_missile_trajectories,inverse_model)

# Scipy的差分进化我们依然保留，用于解决其他问题
from scipy.optimize import differential_evolution

# ===================================================================
# 1. 常量和全局参数定义 (无变化)
# ===================================================================
# ... (这部分和之前完全一样，为了简洁省略) ...
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
# 2. 核心物理模型与目标函数 (无变化)
# ===================================================================
# ... (pre_calculate_missile_trajectories, inverse_model, get_shielding_mask,
#      objective_function_intersection, save_results_to_excel 等函数保持不变) ...

# ===================================================================
# 3. 哈里斯鹰优化器 (HHO) 核心实现
# ===================================================================
class HarrisHawksOptimizer:
    def __init__(self, obj_func, bounds, args, pop_size, max_iter):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.args = args
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        # 初始化鹰群（种群）
        self.positions = np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb
        self.fitness = np.full(self.pop_size, np.inf)

        # 初始化猎物（最优解）
        self.rabbit_position = np.zeros(self.dim)
        self.rabbit_fitness = np.inf

    def levy_flight(self, beta=1.5):
        """生成莱维飞行的步长"""
        # sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
        #          (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma = 0.96567
        u, v = np.random.randn(self.dim), np.random.randn(self.dim)
        step = u * sigma / (np.abs(v) ** (1 / beta))
        return step

    def solve(self):
        print("  - Starting Harris Hawks Optimizer...")
        start_time = time.time()

        for t in range(self.max_iter):
            # 1. 评估每只鹰的适应度
            for i in range(self.pop_size):
                # 确保鹰的位置在边界内
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.positions[i], *self.args)

            # 2. 更新猎物位置（最优解）
            best_hawk_idx = np.argmin(self.fitness)
            if self.fitness[best_hawk_idx] < self.rabbit_fitness:
                self.rabbit_fitness = self.fitness[best_hawk_idx]
                self.rabbit_position = self.positions[best_hawk_idx].copy()

            # 3. 遍历每只鹰，更新其位置
            for i in range(self.pop_size):
                E0 = 2 * np.random.rand() - 1  # 初始能量 (-1, 1)
                E = 2 * E0 * (1 - t / self.max_iter)  # 逃逸能量

                # --- 探索阶段 ---
                if abs(E) >= 1:
                    q = np.random.rand()
                    if q >= 0.5:  # 策略1
                        rand_hawk_idx = np.random.randint(0, self.pop_size)
                        X_rand = self.positions[rand_hawk_idx]
                        self.positions[i] = X_rand - np.random.rand() * np.abs(
                            X_rand - 2 * np.random.rand() * self.positions[i])
                    else:  # 策略2
                        X_m = np.mean(self.positions, axis=0)
                        self.positions[i] = (self.rabbit_position - X_m) - np.random.rand() * (
                                    self.lb + np.random.rand() * (self.ub - self.lb))

                # --- 围捕阶段 ---
                elif abs(E) < 1:
                    r = np.random.rand()  # 成功躲避的概率

                    # 四种策略选择
                    if r >= 0.5 and abs(E) >= 0.5:  # 软包围
                        J = 2 * (1 - np.random.rand())
                        self.positions[i] = (self.rabbit_position - self.positions[i]) - E * np.abs(
                            J * self.rabbit_position - self.positions[i])

                    elif r >= 0.5 and abs(E) < 0.5:  # 硬包围
                        self.positions[i] = self.rabbit_position - E * np.abs(self.rabbit_position - self.positions[i])

                    elif r < 0.5 and abs(E) >= 0.5:  # 渐进式快速俯冲的软包围
                        J = 2 * (1 - np.random.rand())
                        Y = self.rabbit_position - E * np.abs(J * self.rabbit_position - self.positions[i])
                        Z = Y + np.random.rand(self.dim) * self.levy_flight()
                        if self.obj_func(Y, *self.args) < self.fitness[i]:
                            self.positions[i] = Y
                        elif self.obj_func(Z, *self.args) < self.fitness[i]:
                            self.positions[i] = Z

                    elif r < 0.5 and abs(E) < 0.5:  # 渐进式快速俯冲的硬包围
                        J = 2 * (1 - np.random.rand())
                        Y = self.rabbit_position - E * np.abs(
                            J * self.rabbit_position - np.mean(self.positions, axis=0))
                        Z = Y + np.random.rand(self.dim) * self.levy_flight()
                        if self.obj_func(Y, *self.args) < self.fitness[i]:
                            self.positions[i] = Y
                        elif self.obj_func(Z, *self.args) < self.fitness[i]:
                            self.positions[i] = Z

            print(f"  - HHO Iteration {t + 1}/{self.max_iter}, Best Fitness (neg-coverage): {self.rabbit_fitness:.4f}")

        print(f"  - HHO Optimization finished in {time.time() - start_time:.2f} s.")
        return self.rabbit_position, self.rabbit_fitness


# ===================================================================
# 4. 新的问题5求解器 (HHO版本)
# ===================================================================
def solve_problem5_hho(missile_trajectories):
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 5 (using Harris Hawks Optimizer - HHO)")
    print("5 Drones, 3 bombs each, for SIMULTANEOUS shielding of M1, M2, M3")
    print("=" * 50)

    drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
    missile_names = ['M1', 'M2', 'M3']
    bombs_per_drone = 3

    print("  - Building 60-dimensional search space...")
    bounds = []
    max_t_hit = max(missile_trajectories[m]['T_hit'] for m in missile_names)
    for name in drone_names:
        F0 = DRONES_INFO[name]['F0']
        for _ in range(bombs_per_drone):
            bounds.extend([(0, 20000), (-5000, 5000), (0, F0[2] - 1), (1.0, max_t_hit)])
    print("  - Search space built successfully.")

    # 定义HHO参数
    pop_size = 500
    max_iter = 2000

    # 初始化并运行HHO
    hho_optimizer = HarrisHawksOptimizer(
        obj_func=objective_function_intersection_soft_penalty,
        bounds=bounds,
        args=(drone_names, missile_names, missile_trajectories, bombs_per_drone),
        pop_size=pop_size,
        max_iter=max_iter
    )
    best_solution, best_fitness = hho_optimizer.solve()

    if best_fitness < -1e-4:
        max_simultaneous_coverage = -best_fitness
        print(
            f"\n[Problem 5 HHO RESULT] Maximum SIMULTANEOUS Coverage Time (Intersection): {max_simultaneous_coverage:.4f} s")

        all_points = best_solution.reshape(len(drone_names) * bombs_per_drone, 4)
        results_dict = {name: {'strategies': []} for name in drone_names}
        for i, point in enumerate(all_points):
            drone_idx = i // bombs_per_drone
            drone_name = drone_names[drone_idx]
            reach_info = inverse_model(point, DRONES_INFO[drone_name]['F0'])
            if reach_info['is_reachable']:
                strategy = reach_info['strategy']
                strategy['explosion_point'] = point
                results_dict[drone_name]['strategies'].append(strategy)

        save_results_to_excel(results_dict, "result3_hho.xlsx")
    else:
        print("\n[Problem 5 HHO FAILED] Optimization did not find a valid non-zero solution.")


# ===================================================================
# 5. 主程序入口
# ===================================================================
if __name__ == "__main__":
    # ... (前面的问题求解器和准备工作不变) ...
    # 为了演示，我们只运行问题5的HHO版本

    print("Step 1: Pre-calculating all missile trajectories...")
    all_missile_trajectories = pre_calculate_missile_trajectories(MISSILES_INFO)
    print("Done.")

    # 调用新的HHO求解器来解决问题5
    solve_problem5_hho(all_missile_trajectories)

    print("\n\nAll tasks completed.")