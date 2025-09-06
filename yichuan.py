import numpy as np
import random
import pandas as pd
import math
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor
import time

# =========================
# 基础参数和物理模型
# =========================
g = 9.81
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 导弹M1参数
M0 = np.array([20000., 0., 2000.])
u_m = -M0 / np.linalg.norm(M0)


def missile_pos(t):
    return M0 + v_m * t * u_m


# 目标圆柱体参数
TARGET_BASE = np.array([0., 200., 0.])
TARGET_R, TARGET_H = 7.0, 10.0

# 无人机初始位置
FY1_start = np.array([17800., 0., 1800.])


# =========================
# 圆柱体采样点生成（优化版）
# =========================
def sample_cylinder_points_optimized(base, radius, height, n_theta=16, n_z=3):
    """优化的圆柱体采样点生成"""
    pts = []

    # 手动实现linspace功能（避免numba的endpoint参数问题）
    thetas = [i * (2 * np.pi / n_theta) for i in range(n_theta)]
    zs = [i * (height / (n_z - 1)) if n_z > 1 else 0 for i in range(n_z)]

    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0] + radius * c, base[1] + radius * s, base[2] + z])

    # 只采样顶部和底部圆盘的中心区域
    pts.append([base[0], base[1], base[2]])  # 底部中心
    pts.append([base[0], base[1], base[2] + height])  # 顶部中心

    return np.array(pts)


CYL_PTS = sample_cylinder_points_optimized(TARGET_BASE, TARGET_R, TARGET_H)


# =========================
# 距离计算函数（向量化+JIT加速）
# =========================
@jit(nopython=True)
def dist_point_to_segments_batch_fast(P, M, Xs):
    """快速计算点P到线段集[M, Xs]的距离"""
    n = Xs.shape[0]
    min_dist = 1e12

    for i in range(n):
        BA = Xs[i] - M[i]
        BA2 = BA[0] * BA[0] + BA[1] * BA[1] + BA[2] * BA[2]

        if BA2 < 1e-12:
            PA = P - M[i]
            dist = np.sqrt(PA[0] * PA[0] + PA[1] * PA[1] + PA[2] * PA[2])
        else:
            PA = P - M[i]
            dot_product = PA[0] * BA[0] + PA[1] * BA[1] + PA[2] * BA[2]
            s = dot_product / BA2
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0

            Q_x = M[i][0] + s * BA[0]
            Q_y = M[i][1] + s * BA[1]
            Q_z = M[i][2] + s * BA[2]

            dx = P[0] - Q_x
            dy = P[1] - Q_y
            dz = P[2] - Q_z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < min_dist:
            min_dist = dist

    return min_dist


# =========================
# 云团中心计算（优化版）
# =========================
@jit(nopython=True)
def calculate_cloud_center_fast(drone_start, direction, speed, t_drop, tau, t):
    """快速计算烟幕云团中心位置"""
    # 投放点位置
    R_x = drone_start[0] + speed * t_drop * direction[0]
    R_y = drone_start[1] + speed * t_drop * direction[1]
    R_z = drone_start[2] + speed * t_drop * direction[2]

    # 初始速度（水平方向）
    v_x = speed * direction[0]
    v_y = speed * direction[1]
    v_z = speed * direction[2]

    t_exp = t_drop + tau

    # 爆炸点位置（考虑重力）
    E_x = R_x + v_x * tau
    E_y = R_y + v_y * tau
    E_z = R_z + v_z * tau - 0.5 * g * tau * tau

    # 烟幕云下沉后的位置
    dt_sink = t - t_exp
    if dt_sink < 0:
        dt_sink = 0

    C_x = E_x
    C_y = E_y
    C_z = E_z - sink_v * dt_sink

    return np.array([C_x, C_y, C_z])


# =========================
# 问题3适应度函数（优化版）
# =========================
def problem3_fitness_optimized(individual):
    """优化的问题3适应度函数"""
    heading, speed, td1, tau1, td2, tau2, td3, tau3 = individual

    # 约束检查
    drop_times = sorted([td1, td2, td3])
    if min(drop_times[1] - drop_times[0], drop_times[2] - drop_times[1]) < 1.0:
        return -1000.0,

    if speed < 70 or speed > 140:
        return -1000.0,

    direction = np.array([np.cos(heading), np.sin(heading), 0.0])
    te = [td1 + tau1, td2 + tau2, td3 + tau3]
    t_start, t_end = min(te), max(te) + effective_span

    # 优化采样策略
    dt = 0.004  # 增大时间步长
    n_steps = int((t_end - t_start) / dt) + 1
    time_points = np.linspace(t_start, t_end, n_steps)

    cover_time = 0.0

    for t in time_points:
        M_t = missile_pos(t)
        covered = False

        for t_drop, tau in [(td1, tau1), (td2, tau2), (td3, tau3)]:
            if t >= t_drop + tau:
                C = calculate_cloud_center_fast(FY1_start, direction, speed, t_drop, tau, t)

                # 创建重复的导弹位置数组
                M_repeated = np.repeat(M_t.reshape(1, 3), len(CYL_PTS), axis=0)

                min_dist = dist_point_to_segments_batch_fast(C, M_repeated, CYL_PTS)

                if min_dist <= R_cloud:
                    covered = True
                    break

        if covered:
            cover_time += dt

    return cover_time,


# =========================
# 并行适应度计算
# =========================
def evaluate_population_fitness(population):
    """计算种群适应度"""
    results = []
    for ind in population:
        results.append(problem3_fitness_optimized(ind))
    return np.array([r[0] for r in results])


# =========================
# 改进的差分进化算法
# =========================
def improved_differential_evolution():
    """改进的差分进化算法"""
    # 扩展基因范围
    gene_ranges = np.array([
        [0, 2 * np.pi],  # 航向角
        [70, 140],  # 速度（严格限制）
        [0.1, 15.0],  # 投放时间1
        [0.5, 10.0],  # 起爆延迟1
        [0.1, 15.0],  # 投放时间2
        [0.5, 10.0],  # 起爆延迟2
        [0.1, 15.0],  # 投放时间3
        [0.5, 10.0]  # 起爆延迟3
    ])

    pop_size = 50 # 减小种群大小以提高速度
    n_gen = 250 # 减少代数
    F = 0.80
    CR = 0.90

    # 初始化种群
    pop = np.zeros((pop_size, len(gene_ranges)))
    for i in range(pop_size):
        for j in range(len(gene_ranges)):
            pop[i, j] = np.random.uniform(gene_ranges[j, 0], gene_ranges[j, 1])

    # 将用户提供的解作为第一个个体（调整速度到合法范围）
    user_solution = [0.118815, 140.000000, 2.706228, 1.249718, 10.758900, 6.189309, 0.100000, 0.605743]
    pop[0] = np.clip(user_solution, gene_ranges[:, 0], gene_ranges[:, 1])

    # 初始适应度计算
    print("计算初始种群适应度...")
    fitness = evaluate_population_fitness(pop)

    best_fitness_history = []

    for gen in range(n_gen):
        start_time = time.time()

        new_pop = pop.copy()
        new_fitness = fitness.copy()

        for i in range(pop_size):
            # 自适应F参数
            current_F = F * (0.9 + 0.2 * random.random())

            # 变异策略：DE/rand/1
            indices = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = pop[random.sample(indices, 3)]

            mutant = r1 + current_F * (r2 - r3)

            # 边界处理
            for j in range(len(mutant)):
                if mutant[j] < gene_ranges[j, 0]:
                    mutant[j] = gene_ranges[j, 0]
                elif mutant[j] > gene_ranges[j, 1]:
                    mutant[j] = gene_ranges[j, 1]

            # 二项式交叉
            trial = pop[i].copy()
            j_rand = random.randint(0, len(trial) - 1)
            for j in range(len(trial)):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # 评估试验向量
            trial_fitness = problem3_fitness_optimized(trial)[0]

            # 选择
            if trial_fitness > fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = trial_fitness

        pop = new_pop
        fitness = new_fitness

        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_fitness_history.append(best_fitness)

        gen_time = time.time() - start_time
        print(f"Gen {gen}: Best Fitness = {best_fitness:.6f}, Time = {gen_time:.2f}s")
    best_idx = np.argmax(fitness)
    return pop[best_idx], fitness[best_idx]


# =========================
# 主程序
# =========================
# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("开始优化问题3...")
    best_ind3, best_fit3 = improved_differential_evolution()
    print(f"\n优化完成！")
    print(f"最佳遮蔽时间: {best_fit3:.6f}s")

    # 输出最佳参数
    heading, speed, td1, tau1, td2, tau2, td3, tau3 = best_ind3
    print(f"\n最佳参数:")
    print(f"航向角: {heading:.6f} rad")
    print(f"速度: {speed:.6f} m/s")
    print(f"投放时间1: {td1:.6f} s")
    print(f"起爆延迟1: {tau1:.6f} s")
    print(f"投放时间2: {td2:.6f} s")
    print(f"起爆延迟2: {tau2:.6f} s")
    print(f"投放时间3: {td3:.6f} s")
    print(f"起爆延迟3: {tau3:.6f} s")

    # 计算并输出投放点和起爆点信息
    direction = np.array([np.cos(heading), np.sin(heading), 0])
    drops = sorted([(td1, tau1), (td2, tau2), (td3, tau3)], key=lambda x: x[0])

    print(f"\n投放点和起爆点信息:")
    for i, (t_drop, tau) in enumerate(drops, 1):
        # 投放点位置
        drop_pos = FY1_start + speed * t_drop * direction
        # 起爆点位置（爆炸瞬间的位置）
        explosion_pos = calculate_cloud_center_fast(FY1_start, direction, speed, t_drop, tau, t_drop + tau)

        print(f"\n烟幕干扰弹编号 {i}:")
        print(f"  投放时间: {t_drop:.6f} s")
        print(f"  起爆延迟: {tau:.6f} s")
        print(f"  投放点X坐标: {drop_pos[0]:.6f} m")
        print(f"  投放点Y坐标: {drop_pos[1]:.6f} m")
        print(f"  投放点Z坐标: {drop_pos[2]:.6f} m")
        print(f"  起爆点X坐标: {explosion_pos[0]:.6f} m")
        print(f"  起爆点Y坐标: {explosion_pos[1]:.6f} m")
        print(f"  起爆点Z坐标: {explosion_pos[2]:.6f} m")

    # 输出无人机运动信息
    print(f"\n无人机运动信息:")
    print(f"  飞行方向角: {heading:.6f} rad")
    print(f"  飞行速度: {speed:.6f} m/s")

    # 输出最终参数数组
    print(f"\n最终参数数组 [航向角, 速度, 投放时间1, 起爆延迟1, 投放时间2, 起爆延迟2, 投放时间3, 起爆延迟3]:")
    print(f"[{heading:.6f}, {speed:.6f}, {td1:.6f}, {tau1:.6f}, {td2:.6f}, {tau2:.6f}, {td3:.6f}, {tau3:.6f}]")