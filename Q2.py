import math
import numpy as np

# =========================
# 几何/数值工具
# =========================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def dist_point_to_segment(P, A, B):
    """
    点 P 到线段 AB 的距离 + 投影参数 s∈[0,1]
    s≈0 最近点在 A（导弹端）；s≈1 在 B（目标端）；中间值为线段内部。
    """
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
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if fa * fm <= 0:
            hi = mid; fb = fm
        else:
            lo = mid; fa = fm
    return 0.5 * (lo + hi)

def find_cover_intervals(f, t0, t1, dt=0.02):
    """
    在 [t0, t1] 扫描 f(t)=D(t)-R，返回所有满足 D(t)<=R 的时间区间列表 [(tin,tout),...]
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

    intervals = [(a, b) for (a, b) in intervals if b > a + 1e-8]
    return intervals

# =========================
# 题面常量（可按需修改）
# =========================
g = 9.8                 # 重力加速度
v_m = 300.0             # 导弹速度（m/s）
R_cloud = 10.0          # 云团有效半径（m）
sink_v = 3.0            # 云团下沉速度（m/s）
effective_span = 20.0   # 起爆后有效时长（s）

# 真目标与导弹初始（题面给定）
T = np.array([0.0, 200.0, 0.0])          # 真目标下底面圆心
M0 = np.array([20000.0, 0.0, 2000.0])    # M1 初始
u_m = unit(-M0)                           # 直指假目标（原点）的单位方向

# FY1 初始（题面给定）
F0 = np.array([12000.0, 1400.0, 1400.0])    # FY1 初始位置
v_min, v_max = 70.0, 140.0               # FY1 速度范围（m/s）

# =========================
# 轨迹定义
# =========================
def missile_pos(t):
    return M0 + v_m * t * u_m

def drone_pos(t, v_u, theta):
    # 在水平面上等高直线飞行；theta 为朝向角（弧度，x 轴正向逆时针）
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(v_u, theta, t_drop, tau):
    """
    投放点 R 与起爆点 E（只考虑竖直重力，忽略空气阻力；初速即无人机速度方向）
    """
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E

def make_D_minus_R(v_u, theta, t_drop, tau):
    """
    构造当前方案的 D(t)-R_cloud 函数与辅助信息
    """
    R, E = explosion_point(v_u, theta, t_drop, tau)
    t_e = t_drop + tau

    def cloud_center(t):
        return E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)

    def f_scalar(t):
        A = missile_pos(t)
        B = T
        P = cloud_center(t)
        D, _ = dist_point_to_segment(P, A, B)
        return D - R_cloud

    return f_scalar, t_e, R, E

def evaluate_cover_time(v_u, theta, t_drop, tau, scan_dt=0.02):
    """
    返回：总遮蔽时长、遮蔽区间列表、诊断信息（起爆点等）
    约束：起爆高度 E_z>0；速度在 [v_min,v_max]
    """
    if v_u < v_min or v_u > v_max:
        return -1.0, [], None

    # 起爆高度约束（避免在地面/地下起爆）
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0:
        return -1.0, [], None

    f_scalar, t_e, R, E = make_D_minus_R(v_u, theta, t_drop, tau)
    t0, t1 = t_e, t_e + effective_span
    intervals = find_cover_intervals(f_scalar, t0, t1, dt=scan_dt)
    total = sum(b - a for (a, b) in intervals)
    return total, intervals, dict(R=R, E=E, t_e=t_e)

# =========================
# 两阶段优化：粗搜 + 局部精化
# =========================
def coarse_search():
    """
    粗网格搜索：返回若干候选（按总遮蔽时长排序）
    你可以按需调整网格密度以控制速度/精度。
    """
    results = []
    # 航向角：16 个方向（0..2π）
    thetas = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    # 速度档位：70, 90, 110, 130
    v_candidates = np.arange(70,140,1)
    # 投放时刻：0..60 s 每 2 s
    drop_candidates = np.arange(0.0, 12.0 + 1e-9, 1.0)
    # 引信延时：1..9 s 每 1 s（确保 E_z>0；1800-4.9*tau^2>0 允许更大，但此处足够）
    tau_candidates = np.arange(1.0, 12.0 + 1e-9, 1.0)

    for theta in thetas:
        for v_u in v_candidates:
            for t_drop in drop_candidates:
                for tau in tau_candidates:
                    total, intervals, info = evaluate_cover_time(v_u, theta, t_drop, tau, scan_dt=0.03)
                    if total > 0.0:
                        results.append((total, (v_u, theta, t_drop, tau), intervals, info))

    results.sort(key=lambda x: x[0], reverse=True)
    # 返回前 N 个候选用于精化
    return results[:30]

def local_refine(best_list, rng_seed=42, trials_per_seed=200):
    """
    对若干粗搜候选做随机邻域精化搜索（Nelder-Mead 等也可，这里用随机微扰更鲁棒）
    """
    rng = np.random.default_rng(rng_seed)
    incumbents = best_list[:]  # 拷贝一份
    best_overall = incumbents[0] if incumbents else None

    # 从前若干个出发
    for base_idx, base in enumerate(incumbents[:10]):
        base_total, (v0, th0, td0, tau0), _, _ = base
        best_local = base

        for _ in range(trials_per_seed):
            # 以逐步缩小的随机步长微扰
            scale = 1.0 + 0.5 * rng.random()
            v_u  = np.clip(v0  + rng.normal(0, 8.0/scale),  v_min, v_max)
            theta= (th0 + rng.normal(0, (10.0/scale)*math.pi/180.0)) % (2*math.pi)
            t_d  = max(0.0, td0 + rng.normal(0, 2.0/scale))
            tau  = max(0.2, tau0 + rng.normal(0, 0.8/scale))  # 延时>0.2 s，避免数值不稳

            total, intervals, info = evaluate_cover_time(v_u, theta, t_d, tau, scan_dt=0.02)
            if total > best_local[0]:
                best_local = (total, (v_u, theta, t_d, tau), intervals, info)

        # 更新全局
        if best_overall is None or best_local[0] > best_overall[0]:
            best_overall = best_local

    return best_overall

def pretty_angle(theta):
    deg = (theta * 180.0 / math.pi) % 360.0
    return deg

# =========================
# 主流程
# =========================
if __name__ == "__main__":
    print("=== 问题2：自动搜索 FY1 的飞行方向/速度/投放时刻/引信延时，使遮蔽 M1 时间最大化 ===")
    print("阶段1：粗网格搜索 ...")
    coarse = coarse_search()
    if not coarse:
        print("未找到有效遮蔽方案（可能是网格太粗或搜索范围过窄）。")
        exit(0)

    print(f"粗搜 Top-5（总遮蔽时长，速度v，航向角deg，投放t_d，引信τ）：")
    for i, (total, (v_u, theta, t_d, tau), _, _) in enumerate(coarse[:5], 1):
        print(f" {i:2d}. {total:.4f} s | v={v_u:.1f} m/s, θ={pretty_angle(theta):.2f}°, t_d={t_d:.2f} s, τ={tau:.2f} s")

    print("阶段2：局部随机精化 ...")
    best = local_refine(coarse, rng_seed=123, trials_per_seed=250)
    total, (v_u, theta, t_d, tau), intervals, info = best
    R, E, t_e = info["R"], info["E"], info["t_e"]

    print("\n=== 最优方案（经精化） ===")
    print(f"- 总遮蔽时长: {total:.6f} s")
    print(f"- FY1 速度 v: {v_u:.3f} m/s （约束 70~140 m/s）")
    print(f"- FY1 航向角 θ: {pretty_angle(theta):.3f}° （相对 x 轴逆时针）")
    print(f"- 投放时刻 t_d: {t_d:.6f} s")
    print(f"- 引信延时  τ: {tau:.6f} s")
    print(f"- 投放点 R: ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f}) m")
    print(f"- 起爆点 E: ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m （需 >0，当前 E_z={E[2]:.3f}）")
    print(f"- 起爆时刻 t_e: {t_e:.6f} s，评估窗口 [t_e, t_e+20] = [{t_e:.6f}, {t_e+20.0:.6f}] s")

    if intervals:
        for k, (a, b) in enumerate(intervals, 1):
            # 诊断：该区间最小距离与对应 s*（最近点位置类型）
            ts = np.linspace(a, b, 1201)
            Ds, ss = [], []
            for tt in ts:
                A = missile_pos(tt); B = T
                P = E + np.array([0.0, 0.0, -sink_v]) * (tt - t_e)
                D, s = dist_point_to_segment(P, A, B)
                Ds.append(D); ss.append(s)
            imin = int(np.argmin(Ds))
            print(f"  · 区间{k}: [{a:.6f}, {b:.6f}] s，时长 {b-a:.6f} s；"
                  f"最小距离 ≈ {Ds[imin]:.3f} m @ t≈{ts[imin]:.6f} s，s*≈{ss[imin]:.4f}")
    else:
        print("（最优解未形成遮蔽区间——可扩大搜索范围或细化参数网格）")
