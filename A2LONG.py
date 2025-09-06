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
    return [(a,b) for   a,b in intervals if b > a + 1e-8]

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
if __name__ == "__main__":
    MODE = "ALL"   # 可改为 "ALL"

    print(f"=== 问题2（体积目标，{MODE} 口径）：粗网格搜索 ===")
    coarse = coarse_search(mode=MODE)
    if not coarse:
        print("粗搜未找到有效遮蔽方案（可放宽范围或加密网格）。"); exit(0)

    print("粗搜 Top-5：")
    for i,(total,(v_u,th,t_d,tau),_,_) in enumerate(coarse[:5],1):
        print(f"{i:2d}. {total:.4f} s | v={v_u:.1f} m/s, θ={deg(th):.2f}°, t_d={t_d:.2f} s, τ={tau:.2f} s")

    print("\n=== 局部随机精化 ===")
    best = local_refine(coarse, mode=MODE, rng_seed=123, trials_per_seed=300)
    total, (v_u, theta, t_d, tau), intervals, info = best
    R, E, t_e = info["R"], info["E"], info["t_e"]

    print("\n=== 最优方案（体积目标，{} 口径）===".format(MODE))
    print(f"- 总遮蔽时长: {total:.6f} s")
    print(f"- FY1 速度 v: {v_u:.3f} m/s （约束 70~140）")
    print(f"- FY1 航向角 θ: {deg(theta):.3f}° （相对 x 轴逆时针）")
    print(f"- 投放时刻 t_d: {t_d:.6f} s")
    print(f"- 引信延时  τ: {tau:.6f} s")
    print(f"- 投放点 R: ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f}) m")
    print(f"- 起爆点 E: ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m （E_z>0）")
    print(f"- 起爆时刻 t_e: {t_e:.6f} s，评估窗口 [{t_e:.6f}, {t_e+20.0:.6f}] s")

    if intervals:
        # 诊断：区间内取栅格，给出最“紧”的时刻与裕量
        for k,(a,b) in enumerate(intervals,1):
            ts = np.linspace(a, b, 1201)
            vals = []
            for tt in ts:
                M = missile_pos(tt)
                C = cloud_center_builder(E, t_e)(tt)
                Ms = np.repeat(M[None,:], len(CYL_PTS), axis=0)
                d, _ = dist_point_to_segments_batch(C, Ms, CYL_PTS)
                # ANY: 取 min(d)-R； ALL: 取 max(d)-R
                vals.append((np.min(d) if MODE=='ANY' else np.max(d)) - R_cloud)
            imin = int(np.argmin(vals))
            print(f"  · 区间{k}: [{a:.6f}, {b:.6f}] s，时长 {b-a:.6f} s；"
                  f"{'min' if MODE=='ANY' else 'max'}(D-R)≈{vals[imin]:.4f} @ t≈{ts[imin]:.6f} s")
    else:
        print("（最优解未形成遮蔽区间——可扩大/加密搜索范围）")