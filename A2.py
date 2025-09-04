import math
import numpy as np
import tqdm  # 引入tqdm库来显示进度条，非常适合长时间的搜索


# =========================
# 基础向量与几何工具 (代码不变)
# =========================
def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def dist_point_to_segment(P, A, B):
    BA = B - A
    l2 = float(np.dot(BA, BA))
    if l2 == 0.0:
        return float(np.linalg.norm(P - A)), 0.0
    s = float(np.dot(P - A, BA) / l2)
    s_clamped = max(0.0, min(1.0, s))
    Q = A + s_clamped * BA
    return float(np.linalg.norm(P - Q)), s_clamped


def bisect_root(f, a, b, tol=1e-10, maxiter=200):
    fa, fb = f(a), f(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0: return None
    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol: return mid
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5 * (lo + hi)


def find_cover_intervals(f, t0, t1, dt=0.02):
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t)
        vs.append(f(t))
        t += dt
    roots = []
    for i in range(1, len(ts)):
        a, b, fa, fb = ts[i - 1], ts[i], vs[i - 1], vs[i]
        if fa == 0.0: roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None: roots.append(r)
    roots = sorted(roots)
    intervals = []
    inside = lambda t: f(t) <= 0.0
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
    intervals = [(a, b) for (a, b) in intervals if b > a + 1e-8]
    return intervals


# =========================
# 将问题1的求解过程封装成一个函数
# =========================
def calculate_shield_time(h, v_u, t_drop, tau):
    """
    输入一组决策参数，返回总遮蔽时长。
    这是我们优化的目标函数。
    """
    g = 9.8
    v_m = 300.0
    R_cloud = 10.0
    sink_v = 3.0
    effective_span = 20.0
    T = np.array([0.0, 200.0, 0.0])
    M0 = np.array([20000.0, 0.0, 2000.0])
    F0 = np.array([17800.0, 0.0, 1800.0])

    t_e = t_drop + tau
    u_m = unit(-M0)

    missile_pos = lambda t: M0 + v_m * t * u_m
    drone_pos = lambda t: F0 + v_u * t * h

    R = drone_pos(t_drop)
    # 修正平抛运动公式：烟幕弹的初速度是无人机的速度矢量
    v_u_vec = v_u * h
    E = R + v_u_vec * tau + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

    cloud_center = lambda t: E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)

    def D_minus_R(t):
        A = missile_pos(t)
        B = T
        P = cloud_center(t)
        D, s = dist_point_to_segment(P, A, B)
        return D - R_cloud

    t0 = t_e
    t1 = t_e + effective_span

    intervals = find_cover_intervals(D_minus_R, t0, t1, dt=0.05)  # 扫描步长可适当加大以提速
    total_cover = sum(b - a for (a, b) in intervals)

    return total_cover


# =========================
# 问题2：使用网格搜索进行优化
# =========================
if __name__ == "__main__":
    print("=== 第二问：寻找最优投放策略 ===")

    # --- 1. 定义搜索空间和步长 ---
    # 飞行方向(方位角 theta)
    theta_range = np.linspace(0, 2 * np.pi, 37)  # 每10度搜索一次
    # 飞行速度 v_u
    vu_range = np.linspace(70, 140, 8)  # 每10m/s搜索一次
    # 投放时刻 t_drop
    tdrop_range = np.linspace(1.5, 10.0, 18)  # 每0.5s搜索一次
    # 引信延时 tau
    tau_range = np.linspace(1.0, 8.0, 15)  # 每0.5s搜索一次

    # --- 2. 初始化最优解记录 ---
    best_shield_time = -1.0
    best_params = {}

    # --- 3. 执行网格搜索 ---
    total_iterations = len(theta_range) * len(vu_range) * len(tdrop_range) * len(tau_range)
    print(f"开始网格搜索，总计 {total_iterations} 种参数组合...")

    # 使用tqdm创建进度条
    pbar = tqdm.tqdm(total=total_iterations, desc="搜索进度")

    for theta in theta_range:
        for v_u in vu_range:
            for t_drop in tdrop_range:
                for tau in tau_range:
                    # 更新进度条
                    pbar.update(1)

                    # 当前参数组合
                    h = np.array([np.cos(theta), np.sin(theta), 0])

                    # 计算当前组合下的遮蔽时长
                    current_shield_time = calculate_shield_time(h, v_u, t_drop, tau)

                    # 如果找到了更好的解，则更新记录
                    if current_shield_time > best_shield_time:
                        best_shield_time = current_shield_time
                        best_params = {
                            'theta_deg': np.rad2deg(theta),
                            'h': h,
                            'v_u': v_u,
                            't_drop': t_drop,
                            'tau': tau
                        }

    pbar.close()

    # --- 4. 输出最优结果 ---
    print("\n" + "=" * 50)
    print("搜索完成！最优策略如下：")
    print(f"最大有效遮蔽总时长: {best_shield_time:.6f} s")
    print("\n对应的最优参数：")
    print(f"  - 无人机飞行方向角 (theta): {best_params['theta_deg']:.2f} 度")
    print(
        f"  - 无人机飞行方向向量 (h): [{best_params['h'][0]:.4f}, {best_params['h'][1]:.4f}, {best_params['h'][2]:.4f}]")
    print(f"  - 无人机飞行速度 (v_u): {best_params['v_u']:.2f} m/s")
    print(f"  - 烟幕弹投放时刻 (t_drop): {best_params['t_drop']:.2f} s")
    print(f"  - 引信延时 (tau): {best_params['tau']:.2f} s")

    # 根据最优参数，计算并输出投放点和起爆点
    h_opt = best_params['h']
    v_u_opt = best_params['v_u']
    t_drop_opt = best_params['t_drop']
    tau_opt = best_params['tau']

    F0 = np.array([17800.0, 0.0, 1800.0])
    drone_pos_opt = lambda t: F0 + v_u_opt * t * h_opt
    R_opt = drone_pos_opt(t_drop_opt)
    v_u_vec_opt = v_u_opt * h_opt
    E_opt = R_opt + v_u_vec_opt * tau_opt + 0.5 * np.array([0.0, 0.0, -9.8]) * (tau_opt ** 2)

    print("\n根据最优参数计算出的关键点：")
    print(f"  - 烟幕弹投放点 (R): ({R_opt[0]:.2f}, {R_opt[1]:.2f}, {R_opt[2]:.2f})")
    print(f"  - 烟幕弹起爆点 (E): ({E_opt[0]:.2f}, {E_opt[1]:.2f}, {E_opt[2]:.2f})")
    print("=" * 50)