import math
import numpy as np
import random
from deap import base, creator, tools, algorithms
import pandas as pd


# =========================
# 基础工具
# =========================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def bisect_root(f, a, b, tol=1e-10, maxiter=200):
    fa, fb = f(a), f(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0: return None
    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5 * (lo + hi)


def find_cover_intervals(f, t0, t1, dt=0.02):
    """
    在 [t0, t1] 上以步长 dt 扫描 f(t)=D(t)-R，返回所有满足 D(t)<=R 的区间 [(tin,tout),...]
    """
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t)
        vs.append(f(t))
        t += dt

    roots = []
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        fa, fb = vs[i - 1], vs[i]
        if fa == 0.0:
            roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None:
                roots.append(r)
    roots = sorted(roots)

    def inside(t):
        return f(t) <= 0.0

    intervals = []
    cur_in = inside(t0)
    cursor = t0
    for r in roots:
        if cur_in:
            intervals.append((cursor, r))
            cur_in = False
        else:
            cursor = r
            cur_in = True
    if cur_in:
        intervals.append((cursor, t1))

    return [(a, b) for (a, b) in intervals if b > a + 1e-8]


# =========================
# 圆柱体目标采样（半径7，高10）
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0
TARGET_H = 10.0


def sample_cylinder_points(base, radius, height,
                           n_theta_side=64, n_z_side=8,
                           n_theta_disk=48, n_r_disk=3):
    """
    生成圆柱体表面采样点：
    - 侧面：n_theta_side × (n_z_side+1)
    - 上/下底面：极坐标网格（含圆周与若干内环）
    返回 ndarray 形状 (N, 3)
    """
    pts = []

    # 侧面
    thetas = np.linspace(0, 2 * np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0.0, height, n_z_side + 1)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0] + radius * c,
                        base[1] + radius * s,
                        base[2] + z])

    # 上/下底面的极坐标环
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2 * np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk + 1)[1:]  # 去掉 r=0，只取环
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0] + r * c,
                            base[1] + r * s,
                            base[2] + z])

    return np.asarray(pts, dtype=float)


CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)


def dist_point_to_segments_batch(P, Ms, Xs):
    """
    单点 P 到一批线段 [M_i, X_i] 的距离（向量化）
    Ms, Xs: (N,3)
    返回: distances(N,), proj_s(N,)  (投影参数 ∈[0,1])
    """
    BA = Xs - Ms  # (N,3)
    BA2 = np.einsum('ij,ij->i', BA, BA)  # (N,)
    # 处理零长线段的稳健性
    zero = BA2 < 1e-12
    BA2[zero] = 1.0  # 避免除0，稍后用特别分支修正

    PA = P[None, :] - Ms  # (N,3)
    s = np.einsum('ij,ij->i', PA, BA) / BA2
    s = np.clip(s, 0.0, 1.0)
    Q = Ms + s[:, None] * BA  # (N,3)
    d = np.linalg.norm(P[None, :] - Q, axis=1)

    # 零长线段：距离退化为 |P - M|
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1)
        s[zero] = 0.0
    return d, s


# =========================
# 题目物理参数
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 导弹M1（直指假目标原点）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)


def missile_pos(t):
    return M0 + v_m * t * u_m


# FY1 无人机初始位置
F0 = np.array([17800.0, 0.0, 1800.0])

# =========================
# 遗传算法优化
# =========================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 定义基因范围：[航向角, 速度, 投放时间1, 起爆延迟1, 投放时间2, 起爆延迟2, 投放时间3, 起爆延迟3]
LOW = [0, 70, 0, 0, 1.0, 0, 2.0, 0]  # 最小值
UP = [2 * np.pi, 140, 10.0, 6.0, 11.0, 6.0, 12.0, 6.0]  # 最大值


def create_individual():
    """创建个体"""
    individual = [
        random.uniform(LOW[0], UP[0]),  # 航向角
        random.uniform(LOW[1], UP[1]),  # 速度
        random.uniform(LOW[2], UP[2]),  # 投放时间1
        random.uniform(LOW[3], UP[3]),  # 起爆延迟1
        random.uniform(LOW[4], UP[4]),  # 投放时间2
        random.uniform(LOW[5], UP[5]),  # 起爆延迟2
        random.uniform(LOW[6], UP[6]),  # 投放时间3
        random.uniform(LOW[7], UP[7])  # 起爆延迟3
    ]
    return individual


toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def calculate_cloud_center(drone_pos, drop_time, tau, t):
    """计算烟幕云团中心位置"""
    # 投放点
    R = drone_pos(drop_time)
    # 起爆点（考虑重力）
    E = R + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    # 云团中心（考虑下沉）
    t_e = drop_time + tau
    return E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)


def calculate_total_cover_time(individual):
    """计算总遮蔽时间"""
    heading, speed, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3 = individual

    # 最优化的投放时间间隔约束检查
    drop_times = np.array([t_drop1, t_drop2, t_drop3])
    if np.min(np.diff(np.sort(drop_times))) < 1.0:
        return -1000.0

    # 无人机飞行方向向量
    direction = np.array([np.cos(heading), np.sin(heading), 0])

    # 无人机位置函数
    def drone_pos(t):
        return F0 + speed * t * direction

    # 预计算起爆时间和相关参数
    t_e = np.array([t_drop1 + tau1, t_drop2 + tau2, t_drop3 + tau3])
    t_drops = np.array([t_drop1, t_drop2, t_drop3])
    taus = np.array([tau1, tau2, tau3])

    # 按起爆时间排序，便于后续处理
    sort_idx = np.argsort(t_e)
    t_e_sorted = t_e[sort_idx]
    t_drops_sorted = t_drops[sort_idx]
    taus_sorted = taus[sort_idx]

    # 评估时间窗口
    t_start = t_e_sorted[0]
    t_end = t_e_sorted[-1] + effective_span

    def combined_f(t):
        """组合三个烟幕弹的遮蔽效果"""
        min_distance = float('inf')
        M_t = missile_pos(t)
        Ms_expanded = np.repeat(M_t[None, :], len(CYL_PTS), axis=0)

        # 按起爆时间顺序检查
        for i in range(3):
            if t >= t_e_sorted[i]:
                C = calculate_cloud_center(drone_pos, t_drops_sorted[i], taus_sorted[i], t)
                d = dist_point_to_segments_batch(C, Ms_expanded, CYL_PTS)[0]  # 只取距离
                current_min = np.min(d)
                if current_min < min_distance:
                    min_distance = current_min
                    # 如果已经找到遮蔽点，可以提前终止
                    if min_distance <= R_cloud:
                        break
            else:
                # 如果当前烟幕弹还未起爆，后面的也不会起爆（因为已排序）
                break

        return min_distance - R_cloud

    # 计算遮蔽区间
    intervals = find_cover_intervals(combined_f, t_start, t_end, dt=0.1)
    total_time = sum(b - a for a, b in intervals)

    return total_time

def evaluate(individual):
    """适应度函数"""
    total_time = calculate_total_cover_time(individual)
    return total_time,


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    """主遗传算法流程"""
    # 参数设置
    population_size = 50
    n_generations = 30
    cx_prob = 0.5
    mut_prob = 0.5

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 运行遗传算法
    population, logbook = algorithms.eaSimple(
        population, toolbox, cx_prob, mut_prob, n_generations,
        stats=stats, verbose=True
    )

    # 选择最佳个体
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"\n最佳适应度: {best_fitness:.6f}")
    print(f"最佳参数: {best_individual}")

    # 解析最佳个体
    heading, speed, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3 = best_individual
    direction = np.array([np.cos(heading), np.sin(heading), 0])

    # 保存结果到Excel
    results = []
    for i, (t_drop, tau) in enumerate([(t_drop1, tau1), (t_drop2, tau2), (t_drop3, tau3)], 1):
        drop_pos = F0 + speed * t_drop * direction
        results.append({
            '烟幕弹编号': i,
            '投放时间(s)': t_drop,
            '起爆延迟(s)': tau,
            '投放点X(m)': drop_pos[0],
            '投放点Y(m)': drop_pos[1],
            '投放点Z(m)': drop_pos[2]
        })

    df = pd.DataFrame(results)
    df.to_excel('result1.xlsx', index=False)
    print("结果已保存到 result1.xlsx")

    # 输出无人机信息
    print(f"\n无人机FY1策略:")
    print(f"飞行方向角: {heading:.6f} rad")
    print(f"飞行速度: {speed:.6f} m/s")

    return best_individual, best_fitness


def main():
    """主遗传算法流程"""
    # 参数设置
    population_size = 50
    n_generations = 30
    cx_prob = 0.5
    mut_prob = 0.5

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 运行遗传算法
    population, logbook = algorithms.eaSimple(
        population, toolbox, cx_prob, mut_prob, n_generations,
        stats=stats, verbose=True
    )

    # 选择最佳个体
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"\n最佳适应度: {best_fitness:.6f}")
    print(f"最佳参数: {best_individual}")

    # 解析最佳个体
    heading, speed, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3 = best_individual
    direction = np.array([np.cos(heading), np.sin(heading), 0])

    # 计算无人机飞行方向（角度制）
    heading_deg = np.degrees(heading)

    # 计算各烟幕弹的投放点和起爆点
    results = []
    for i, (t_drop, tau) in enumerate([(t_drop1, tau1), (t_drop2, tau2), (t_drop3, tau3)], 1):
        # 投放点
        drop_pos = F0 + speed * t_drop * direction

        # 起爆点（考虑重力影响）
        explosion_pos = drop_pos + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

        results.append({
            '烟幕弹编号': i,
            '投放时间(s)': t_drop,
            '起爆延迟(s)': tau,
            '投放点X(m)': drop_pos[0],
            '投放点Y(m)': drop_pos[1],
            '投放点Z(m)': drop_pos[2],
            '起爆点X(m)': explosion_pos[0],
            '起爆点Y(m)': explosion_pos[1],
            '起爆点Z(m)': explosion_pos[2]
        })

    # 保存结果到Excel
    df = pd.DataFrame(results)
    df.to_excel('result1.xlsx', index=False)
    print("结果已保存到 result1.xlsx")

    # 输出无人机信息
    print(f"\n无人机FY1策略:")
    print(f"飞行方向角: {heading:.6f} rad ({heading_deg:.2f}°)")
    print(f"飞行速度: {speed:.6f} m/s")

    # 输出各烟幕弹详细信息
    print(f"\n烟幕干扰弹投放信息:")
    for i, result in enumerate(results, 1):
        print(f"\n烟幕弹 {i}:")
        print(f"  投放时间: {result['投放时间(s)']:.6f} s")
        print(f"  起爆延迟: {result['起爆延迟(s)']:.6f} s")
        print(f"  投放点: ({result['投放点X(m)']:.2f}, {result['投放点Y(m)']:.2f}, {result['投放点Z(m)']:.2f}) m")
        print(f"  起爆点: ({result['起爆点X(m)']:.2f}, {result['起爆点Y(m)']:.2f}, {result['起爆点Z(m)']:.2f}) m")

    # 计算并输出总遮蔽时间
    total_time = calculate_total_cover_time(best_individual)
    print(f"\n总有效遮蔽时间: {total_time:.6f} s")

    # 详细分析遮蔽时间段
    print(f"\n遮蔽时间段分析:")
    heading, speed, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3 = best_individual
    direction = np.array([np.cos(heading), np.sin(heading), 0])

    def drone_pos(t):
        return F0 + speed * t * direction

    t_e = np.array([t_drop1 + tau1, t_drop2 + tau2, t_drop3 + tau3])
    t_drops = np.array([t_drop1, t_drop2, t_drop3])
    taus = np.array([tau1, tau2, tau3])

    sort_idx = np.argsort(t_e)
    t_e_sorted = t_e[sort_idx]
    t_drops_sorted = t_drops[sort_idx]
    taus_sorted = taus[sort_idx]

    t_start = t_e_sorted[0]
    t_end = t_e_sorted[-1] + effective_span

    def combined_f(t):
        min_distance = float('inf')
        M_t = missile_pos(t)
        Ms_expanded = np.repeat(M_t[None, :], len(CYL_PTS), axis=0)

        for i in range(3):
            if t >= t_e_sorted[i]:
                C = calculate_cloud_center(drone_pos, t_drops_sorted[i], taus_sorted[i], t)
                d = dist_point_to_segments_batch(C, Ms_expanded, CYL_PTS)[0]
                current_min = np.min(d)
                if current_min < min_distance:
                    min_distance = current_min
                    if min_distance <= R_cloud:
                        break
            else:
                break
        return min_distance - R_cloud

    intervals = find_cover_intervals(combined_f, t_start, t_end, dt=0.1)

    print(f"发现 {len(intervals)} 个遮蔽时间段:")
    for i, (start, end) in enumerate(intervals, 1):
        duration = end - start
        print(f"  时间段 {i}: {start:.2f} s - {end:.2f} s (持续时间: {duration:.2f} s)")

    return best_individual, best_fitness

if __name__ =='__main__':
    main()