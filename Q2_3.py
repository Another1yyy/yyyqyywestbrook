import math, numpy as np, cma, joblib
from numba import njit, prange
from joblib import Parallel, delayed
import math
import numpy as np

# =========================
# 通用工具
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
        mid = 0.5*(lo+hi); fm = f(mid)
        if abs(fm) < tol or (hi-lo) < tol: return mid
        if fa * fm <= 0: hi, fb = mid, fm
        else: lo, fa = mid, fm
    return 0.5*(lo+hi)

def find_cover_intervals(f, t0, t1, dt=0.03):
    """在 [t0,t1] 以步长 dt 扫描 f(t)=D(t)-R，返回所有满足 D<=R 的区间列表"""
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t); vs.append(f(t)); t += dt

    roots = []
    for i in range(1, len(ts)):
        a, b = ts[i-1], ts[i]
        fa, fb = vs[i-1], vs[i]
        if fa == 0.0: roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b);  roots.append(r) if r is not None else None
    roots = sorted(roots)

    def inside(x): return f(x) <= 0.0
    intervals = []
    cur_in, cursor = inside(t0), t0
    for r in roots:
        if cur_in:
            intervals.append((cursor, r)); cur_in = False
        else:
            cursor, cur_in = r, True
    if cur_in: intervals.append((cursor, t1))
    return [(a,b) for a,b in intervals if b > a + 1e-8]

# =========================
# “体积目标”——圆柱体采样
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0
TARGET_H = 10.0

def sample_cylinder_points(base, radius, height,
                           n_theta_side=64, n_z_side=8,
                           n_theta_disk=48, n_r_disk=3):
    pts = []
    thetas = np.linspace(0, 2*np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0.0, height, n_z_side+1)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0]+radius*c, base[1]+radius*s, base[2]+z])
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]  # 去 r=0
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0]+r*c, base[1]+r*s, base[2]+z])
    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    """单点 P 到一批线段 [M_i, X_i] 的距离（向量化）"""
    BA = Xs - Ms                      # (N,3)
    BA2 = np.einsum('ij,ij->i', BA, BA)
    zero = BA2 < 1e-12
    BA2[zero] = 1.0
    PA = P[None, :] - Ms
    s = np.einsum('ij,ij->i', PA, BA) / BA2
    s = np.clip(s, 0.0, 1.0)
    Q = Ms + s[:, None] * BA
    d = np.linalg.norm(P[None, :] - Q, axis=1)
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1)
        s[zero] = 0.0
    return d, s

# =========================
# 题面参数（导弹/无人机/云团）
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 导弹：直指假目标原点
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

# FY1 初始
F0 = np.array([17800.0, 0.0, 1800.0])
v_min, v_max = 70.0, 140.0

# 云团中心（由起爆点 E、起爆时刻 t_e 决定）
def cloud_center_builder(E, t_e):
    def C(t): return E + np.array([0.0,0.0,-sink_v]) * (t - t_e)
    return C

# 体积目标遮蔽判定函数构造器
def build_volume_cover_fn(C_pts, missile_pos_func, cloud_center_func, mode='ANY'):
    Xi = C_pts
    if mode.upper() == 'ANY':
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.min(d) - R_cloud)
        return f
    else:  # 'ALL'
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

# 轨迹与起爆点
def drone_pos(t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5 * np.array([0.0,0.0,-g]) * (tau**2)
    return R, E

# 单次方案评估
def evaluate_cover_time(v_u, theta, t_drop, tau, mode='ANY', scan_dt=0.03):
    if not (v_min <= v_u <= v_max): return -1.0, [], None
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return -1.0, [], None   # 起爆高度>0
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    intervals = find_cover_intervals(f, t_e, t_e + effective_span, dt=scan_dt)
    total = sum(b-a for a,b in intervals)
    info = dict(R=drone_pos(t_drop, v_u, theta), E=E, t_e=t_e)
    return total, intervals, info

# =========================
# 粗搜 + 局部精化
# =========================
def coarse_search(mode='ANY'):
    results = []
    thetas = np.linspace(0.0, 2.0*np.pi, 16, endpoint=False)
    v_candidates = [70.0, 90.0, 110.0, 130.0]
    drop_candidates = np.arange(0.0, 60.0+1e-9, 2.0)
    tau_candidates  = np.arange(1.0, 10.0+1e-9, 1.0)

    for theta in thetas:
        for v_u in v_candidates:
            for t_d in drop_candidates:
                for tau in tau_candidates:
                    total, intervals, info = evaluate_cover_time(
                        v_u, theta, t_d, tau, mode=mode, scan_dt=0.04
                    )
                    if total > 0.0:
                        results.append((total, (v_u, theta, t_d, tau), intervals, info))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:30]

def local_refine(best_list, mode='ANY', rng_seed=42, trials_per_seed=250):
    rng = np.random.default_rng(rng_seed)
    if not best_list: return None
    best_overall = best_list[0]

    for base in best_list[:10]:
        base_total, (v0, th0, td0, tau0), _, _ = base
        best_local = base
        for _ in range(trials_per_seed):
            scale = 1.0 + 0.5 * rng.random()
            v_u  = np.clip(v0 + rng.normal(0, 8.0/scale), v_min, v_max)
            theta= (th0 + rng.normal(0, (10.0/scale)*math.pi/180.0)) % (2*np.pi)
            t_d  = max(0.0, td0 + rng.normal(0, 2.0/scale))
            tau  = max(0.2, tau0 + rng.normal(0, 0.8/scale))

            total, intervals, info = evaluate_cover_time(
                v_u, theta, t_d, tau, mode=mode, scan_dt=0.02
            )
            if total > best_local[0]:
                best_local = (total, (v_u, theta, t_d, tau), intervals, info)
        if best_local[0] > best_overall[0]:
            best_overall = best_local
    return best_overall

def deg(theta): return (theta * 180.0 / math.pi) % 360.0

# =========================
# 主程序
# =========================

# ---------- 1. Numba 加速核心函数 ----------
@njit(fastmath=True, parallel=True)
def min_dist_point_segments(P, Ms, Xs):
    """
    P : (3,)
    Ms: (N,3) 线段起点
    Xs: (N,3) 线段终点
    返回最小距离
    """
    N = Ms.shape[0]
    min_d = 1e99
    for i in prange(N):
        A = Ms[i]
        B = Xs[i]
        AB = B - A
        AB2 = AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]
        if AB2 < 1e-16:
            d2 = (P[0]-A[0])**2 + (P[1]-A[1])**2 + (P[2]-A[2])**2
        else:
            AP = P - A
            t = (AP[0]*AB[0] + AP[1]*AB[1] + AP[2]*AB[2]) / AB2
            t = max(0., min(1., t))
            Q = A + t * AB
            d2 = (P[0]-Q[0])**2 + (P[1]-Q[1])**2 + (P[2]-Q[2])**2
        if d2 < min_d:
            min_d = d2
    return math.sqrt(min_d)

# ---------- 2. 遮蔽函数 ----------
def build_volume_cover_fn_fast(C_pts, missile_pos_func, cloud_center_func, mode='ANY'):
    Xi = C_pts
    if mode.upper() == 'ANY':
        def f(t):
            M = missile_pos_func(t)
            C = cloud_center_func(t)
            return min_dist_point_segments(C, M.reshape(1,3).repeat(len(Xi),0), Xi) - R_cloud
        return f
    else:
        def f(t):
            M = missile_pos_func(t)
            C = cloud_center_func(t)
            d = min_dist_point_segments(C, M.reshape(1,3).repeat(len(Xi),0), Xi)
            return d - R_cloud   # 仅示例，ALL 模式需改写为 max
        return f

# ---------- 3. 快速区间扫描（预筛 + 二分） ----------
def find_cover_intervals_fast(f, t0, t1, dt_coarse=0.2, dt_fine=0.03):
    # 1. 预筛
    ts = np.arange(t0, t1+1e-9, dt_coarse)
    vs = np.asarray([f(t) for t in ts])
    flag = vs <= 0
    # 2. 精细二分
    intervals = []
    for i in range(len(ts)-1):
        if flag[i] or flag[i+1] or (vs[i]*vs[i+1] < 0):
            a,b = ts[i], ts[i+1]
            roots = []
            fa, fb = vs[i], vs[i+1]
            if fa*fb < 0:
                roots.append(bisect_root(f, a, b))
            # 子区间内部再细扫
            tss = np.arange(a, b+1e-9, dt_fine)
            vss = [f(t) for t in tss]
            inside = False
            for j in range(len(tss)):
                if vss[j] <= 0:
                    if not inside:
                        start = tss[j]; inside = True
                else:
                    if inside:
                        intervals.append((start, tss[j])); inside = False
            if inside:
                intervals.append((start, tss[-1]))
    return [(a,b) for a,b in intervals if b>a+1e-8]

# ---------- 4. 多进程粗搜 ----------
def coarse_search_parallel(mode='ANY', n_jobs=-1):
    thetas = np.linspace(0, 2*np.pi, 16, endpoint=False)
    v_list = [70., 90., 110., 130.]
    td_list = np.arange(0, 60.1, 2)
    tau_list = np.arange(1., 10.1, 1)

    def task(args):
        v_u, theta, t_d, tau = args
        total, intervals, info = evaluate_cover_time_fast(v_u, theta, t_d, tau, mode)
        return (total, (v_u, theta, t_d, tau), intervals, info) if total>0 else None

    tasks = [(v,th,td,tau) for th in thetas for v in v_list for td in td_list for tau in tau_list]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(task)(t) for t in tasks)
    res = [r for r in res if r is not None]
    res.sort(key=lambda x: x[0], reverse=True)
    return res[:30]

# ---------- 5. CMA-ES 精化 ----------
def refine_cma(best_list, mode='ANY'):
    if not best_list: return None
    base = best_list[0]
    total0, (v0,th0,td0,tau0), _, _ = base
    es = cma.CMAEvolutionStrategy([v0, th0, td0, tau0],
                                  0.15*np.array([20, 0.3, 10, 2]),
                                  {'bounds': [[70,0,0,0.2], [140,2*np.pi,60,10]]})
    best = base
    while not es.stop():
        solutions = es.ask()
        values = [-evaluate_cover_time_fast(v,th,td,tau,mode)[0] for v,th,td,tau in solutions]
        es.tell(solutions, values)
        idx = np.argmin(values)
        if -values[idx] > best[0]:
            best = (-values[idx], solutions[idx], [], None)
    return best

# ---------- 6. 与原脚本兼容的 evaluate_cover_time_fast ----------
def evaluate_cover_time_fast(v_u, theta, t_drop, tau, mode='ANY'):
    if not (v_min <= v_u <= v_max): return -1.0, [], None
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0: return -1.0, [], None
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn_fast(CYL_PTS, missile_pos, C, mode)
    intervals = find_cover_intervals_fast(f, t_e, t_e+effective_span)
    total = sum(b-a for a,b in intervals)
    info = dict(R=drone_pos(t_drop, v_u, theta), E=E, t_e=t_e)
    return total, intervals, info

# ---------- 7. 主程序 ----------
if __name__ == '__main__':
    MODE = 'ANY'
    print('=== 多进程粗搜 + CMA 精化 ===')
    coarse = coarse_search_parallel(mode=MODE, n_jobs=16)   # 按需改核数
    best = refine_cma(coarse, mode=MODE)
    total, (v_u, theta, t_d, tau), intervals, info = best
    R, E, t_e = info['R'], info['E'], info['t_e']
    print(f'最优遮蔽时长: {total:.6f} s')
    print(f'v={v_u:.3f} m/s, θ={deg(theta):.2f}°, t_d={t_d:.3f} s, τ={tau:.3f} s')