# -*- coding: utf-8 -*-
"""
Problem 4 via Particle Swarm Optimization (PSO)
- Two UAVs (FY1, FY2) collaborate
- Maximize union cover time with total number of drops limited (N_TOTAL)
- Volume (cylindrical) target occlusion, ANY/ALL modes

Dependencies: numpy, math
"""
import math
import csv
import numpy as np

# =========================
# 可调参数
# =========================
MODE = "ANY"               # "ANY" or "ALL"
SEED = 42                  # 随机种子
SWARM_SIZE = 80            # 粒子数
MAX_ITERS  = 140           # 迭代次数
W_START, W_END = 0.8, 0.4  # 惯性权重线性递减
C1, C2 = 1.6, 1.6          # 个体/群体学习因子

# 总投放数（两机合计）
N_TOTAL = 6
# 同一无人机相邻投放最小间隔（秒）
MIN_GAP = 1.0

# 时间离散：粗评与精评
SCAN_DT_COARSE = 0.06
SCAN_DT_FINE   = 0.02

# 如需保存为 CSV
SAVE_CSV = False

# =========================
# 几何/数值工具
# =========================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def bisect_root(f, a, b, tol=1e-10, maxiter=120):
    fa, fb = f(a), f(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0: return None
    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if abs(fm) < tol or (hi-lo) < tol:
            return mid
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5*(lo+hi)

def find_cover_intervals(f, t0, t1, dt=0.06):
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
            r = bisect_root(f, a, b)
            if r is not None: roots.append(r)
    roots = sorted(roots)
    def inside(x): return f(x) <= 0.0
    ivs, cur_in, cursor = [], inside(t0), t0
    for r in roots:
        if cur_in: ivs.append((cursor, r)); cur_in = False
        else: cursor, cur_in = r, True
    if cur_in: ivs.append((cursor, t1))
    return [(a,b) for a,b in ivs if b > a + 1e-8]

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9: out[-1] = (la, max(lb, b))
        else: out.append((a,b))
    return out

def total_length(intervals):
    return sum(b-a for a,b in intervals)

def deg(rad): return (rad * 180.0 / math.pi) % 360.0

# =========================
# 体积目标（圆柱）采样（性能/精度折中）
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0
TARGET_H = 10.0

def sample_cylinder_points(base, radius, height,
                           n_theta_side=40, n_z_side=6,
                           n_theta_disk=24, n_r_disk=2):
    pts = []
    thetas = np.linspace(0, 2*np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0.0, height, n_z_side+1)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0]+radius*c, base[1]+radius*s, base[2]+z])
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0]+r*c, base[1]+r*s, base[2]+z])
    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    BA  = Xs - Ms
    BA2 = np.einsum('ij,ij->i', BA, BA)
    zero = BA2 < 1e-12
    BA2[zero] = 1.0
    PA = P[None, :] - Ms
    s  = np.einsum('ij,ij->i', PA, BA) / BA2
    s  = np.clip(s, 0.0, 1.0)
    Q  = Ms + s[:, None] * BA
    d  = np.linalg.norm(P[None, :] - Q, axis=1)
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1)
        s[zero] = 0.0
    return d, s

# =========================
# 题面物理（导弹/无人机/云团）
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 导弹（直指假目标原点）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t): return M0 + v_m * t * u_m

# 两架无人机初始（如题面不同请改这里）
FY1_F0 = np.array([17800.0, 0.0, 1800.0])
FY2_F0 = np.array([17800.0, 800.0, 1800.0])

# 变量边界
THETA_B = (0.0, 2.0*math.pi)  # rad
V_B     = (70.0, 140.0)       # m/s
TD_B    = (0.0, 60.0)         # s
TAU_B   = (0.25, 9.0)         # s
A_B     = (0.0, 1.0)          # 分配开关（0→FY1, 1→FY2）

def drone_pos(F0, t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(F0, v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(F0, t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5*np.array([0.0,0.0,-g])*(tau**2)
    return R, E

def cloud_center_builder(E, t_e):
    def C(t): return E + np.array([0.0,0.0,-sink_v])*(t - t_e)
    return C

def build_volume_cover_fn(Xi, missile_pos_func, cloud_center_func, mode='ANY'):
    Xi = np.asarray(Xi)
    if mode.upper() == 'ANY':
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.min(d) - R_cloud)
        return f
    else:
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

def evaluate_single(F0, v_u, theta, t_drop, tau, dt=SCAN_DT_COARSE):
    R, E = explosion_point(F0, v_u, theta, t_drop, tau)
    if E[2] <= 0.0:  # 起爆高度>0
        return [], dict(R=R, E=E, t_e=t_drop+tau)
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=MODE)
    ivs = find_cover_intervals(f, t_e, t_e + effective_span, dt=dt)
    return ivs, dict(R=R, E=E, t_e=t_e)

# =========================
# 间隔约束处理（每机独立）
# =========================
def enforce_times_seq(times, tmin=TD_B[0], tmax=TD_B[1], gap=MIN_GAP):
    """
    给定一个投放时刻列表（某一架无人机），
    返回按时间排序且满足最小间隔的时刻序列；若时间窗不足则自适应压缩。
    """
    n = len(times)
    if n <= 1:
        return [float(np.clip(times[0], tmin, tmax))] if n == 1 else []
    times = sorted([float(t) for t in times])
    # 若 gap 过大使得不可行，降到极限（留1%裕度）
    max_gap = (tmax - tmin) / (n - 1)
    g = min(gap, 0.99 * max_gap) if max_gap > 0 else 0.0
    # 先把第一个压到允许位置
    times[0] = float(np.clip(times[0], tmin, tmax - (n-1)*g))
    for i in range(1, n):
        times[i] = max(times[i], times[i-1] + g)
    # 右边越界则整体左移
    excess = times[-1] - tmax
    if excess > 0:
        times = [t - excess for t in times]
    # 左边越界则整体右移
    deficit = tmin - times[0]
    if deficit > 0:
        times = [t + deficit for t in times]
    # 最终裁剪
    times = [float(np.clip(t, tmin, tmax)) for t in times]
    # 再次保证间隔（防浮点）
    for i in range(1, n):
        if times[i] < times[i-1] + g:
            times[i] = min(times[i-1] + g, tmax)
    return times

# =========================
# 评价一个“联合投放计划”
# x = [v1, th1, v2, th2, t1,tau1,a1, ..., tN, tauN, aN]
# =========================
def evaluate_plan(x, dt=SCAN_DT_COARSE):
    v1, th1, v2, th2 = x[0], x[1], x[2], x[3]
    # 变量边界内裁剪 + 角度归一
    v1 = float(np.clip(v1, V_B[0], V_B[1])); th1 = float(np.mod(th1, 2.0*math.pi))
    v2 = float(np.clip(v2, V_B[0], V_B[1])); th2 = float(np.mod(th2, 2.0*math.pi))

    # 解析各枚
    drops1, drops2 = [], []
    for k in range(N_TOTAL):
        t   = np.clip(x[4 + 3*k + 0], TD_B[0], TD_B[1])
        tau = np.clip(x[4 + 3*k + 1], TAU_B[0], TAU_B[1])
        a   = np.clip(x[4 + 3*k + 2], A_B[0],  A_B[1])
        uav = 1 if a >= 0.5 else 0  # 0 → FY1, 1 → FY2
        if uav == 0:
            drops1.append([float(t), float(tau)])
        else:
            drops2.append([float(t), float(tau)])

    # 各机分别排序并施加间隔约束
    T1 = enforce_times_seq([d[0] for d in drops1], gap=MIN_GAP)
    T2 = enforce_times_seq([d[0] for d in drops2], gap=MIN_GAP)
    for i in range(len(drops1)): drops1[i][0] = T1[i]
    for i in range(len(drops2)): drops2[i][0] = T2[i]

    # 评估区间并取并集
    ivs_all = []
    # FY1
    for t, tau in drops1:
        ivs, _ = evaluate_single(FY1_F0, v1, th1, t, tau, dt=dt)
        ivs_all += ivs
    # FY2
    for t, tau in drops2:
        ivs, _ = evaluate_single(FY2_F0, v2, th2, t, tau, dt=dt)
        ivs_all += ivs

    merged = merge_intervals(ivs_all)
    return total_length(merged), merged, (v1, th1, v2, th2), (drops1, drops2)

# =========================
# 粒子群优化（维度 = 4 + 3*N_TOTAL）
# =========================
def pso_optimize(seed=SEED):
    rng = np.random.default_rng(seed)
    dim = 4 + 3*N_TOTAL
    LB = np.array([V_B[0], THETA_B[0], V_B[0], THETA_B[0]] +
                  [TD_B[0], TAU_B[0], A_B[0]]*N_TOTAL, dtype=float)
    UB = np.array([V_B[1], THETA_B[1], V_B[1], THETA_B[1]] +
                  [TD_B[1], TAU_B[1], A_B[1]]*N_TOTAL, dtype=float)

    # 初始化
    X = rng.uniform(LB, UB, size=(SWARM_SIZE, dim))
    # 粒子速度（按变量尺度初始化）
    V = rng.uniform(-1.0, 1.0, size=(SWARM_SIZE, dim)) * (UB - LB) * 0.1
    # 个体与全局最优（PSO最小化：用 -total_time）
    def fitness(vec, dt=SCAN_DT_COARSE):
        tot, _, _, _ = evaluate_plan(vec, dt=dt)
        return -max(tot, 0.0)

    pbestX = X.copy()
    pbestF = np.array([fitness(x, dt=SCAN_DT_COARSE) for x in X])
    gidx = int(np.argmin(pbestF))
    gbestX = pbestX[gidx].copy()
    gbestF = float(pbestF[gidx])

    # 速度夹限（经验），避免发散
    vmin = np.full(dim, -6.0); vmax = np.full(dim, 6.0)
    vmin[0] = vmin[2] = -20.0; vmax[0] = vmax[2] = 20.0   # 速度变量
    vmin[1] = vmin[3] = -math.pi; vmax[1] = vmax[3] = math.pi  # 角速度

    for it in range(1, MAX_ITERS+1):
        w = W_START + (W_END - W_START) * (it / MAX_ITERS)
        r1 = rng.random(size=X.shape)
        r2 = rng.random(size=X.shape)
        V = w*V + C1*r1*(pbestX - X) + C2*r2*(gbestX - X)
        V = np.maximum(np.minimum(V, vmax), vmin)

        X = X + V
        # 角度 wrap + 边界裁剪
        X[:,0] = np.clip(X[:,0], V_B[0], V_B[1])
        X[:,2] = np.clip(X[:,2], V_B[0], V_B[1])
        X[:,1] = np.mod(X[:,1], 2.0*math.pi)
        X[:,3] = np.mod(X[:,3], 2.0*math.pi)
        # 其余变量裁剪
        for k in range(N_TOTAL):
            base = 4 + 3*k
            X[:, base+0] = np.clip(X[:, base+0], TD_B[0], TD_B[1])  # t
            X[:, base+1] = np.clip(X[:, base+1], TAU_B[0], TAU_B[1])# tau
            X[:, base+2] = np.clip(X[:, base+2], A_B[0],  A_B[1])   # a

        # 更新个体/全局最优
        for i in range(SWARM_SIZE):
            f = fitness(X[i], dt=SCAN_DT_COARSE)
            if f < pbestF[i]:
                pbestF[i] = f; pbestX[i] = X[i].copy()
                if f < gbestF:
                    gbestF = f; gbestX = X[i].copy()

        if it % 10 == 0 or it == 1 or it == MAX_ITERS:
            print(f"[PSO-4] iter {it:3d} | best(coarse) ≈ {-gbestF:.4f} s")

    # 高精复核
    best_total, merged, (v1, th1, v2, th2), (drops1, drops2) = evaluate_plan(gbestX, dt=SCAN_DT_FINE)
    return best_total, merged, (v1, th1, v2, th2), (drops1, drops2)

# =========================
# 输出与可选保存
# =========================
def save_csv(prefix, best_total, merged, params, drops1, drops2):
    v1, th1, v2, th2 = params
    with open(prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uav","theta_deg","v_mps","mode","total_cover_s"])
        w.writerow(["FY1", f"{deg(th1):.6f}", f"{v1:.6f}", MODE, f"{best_total:.6f}"])
        w.writerow(["FY2", f"{deg(th2):.6f}", f"{v2:.6f}", MODE, ""])

    with open(prefix + "_plan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uav","t_drop_s","tau_s"])
        for t, tau in drops1: w.writerow(["FY1", f"{t:.6f}", f"{tau:.6f}"])
        for t, tau in drops2: w.writerow(["FY2", f"{t:.6f}", f"{tau:.6f}"])

    with open(prefix + "_intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tin_s","tout_s","duration_s"])
        for a,b in merged:
            w.writerow([f"{a:.6f}", f"{b:.6f}", f"{(b-a):.6f}"])

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"=== Problem 4 via PSO (MODE={MODE}, N_TOTAL={N_TOTAL}, MIN_GAP={MIN_GAP}) ===")
    best_total, merged, (v1, th1, v2, th2), (drops1, drops2) = pso_optimize(seed=SEED)

    print("\n=== 最优方案（PSO，高精复核） ===")
    print(f"- 并集总时长: {best_total:.6f} s")
    print(f"- FY1: θ={deg(th1):.3f}°, v={v1:.3f} m/s, 计划投放 {len(drops1)} 枚")
    for i,(t,tau) in enumerate(sorted(drops1), 1):
        print(f"   · FY1#{i}: t={t:.6f} s, τ={tau:.6f} s")
    print(f"- FY2: θ={deg(th2):.3f}°, v={v2:.3f} m/s, 计划投放 {len(drops2)} 枚")
    for i,(t,tau) in enumerate(sorted(drops2), 1):
        print(f"   · FY2#{i}: t={t:.6f} s, τ={tau:.6f} s")
    if merged:
        for k,(a,b) in enumerate(merged,1):
            print(f"  · 并集区间{k}: [{a:.6f}, {b:.6f}] s，时长 {(b-a):.6f} s")
    else:
        print("（未形成遮蔽并集；建议增大粒子数/迭代或减小 SCAN_DT_FINE）")

    if SAVE_CSV:
        save_csv("p4_pso", best_total, merged, (v1, th1, v2, th2), drops1, drops2)
        print("[OK] 结果已写 CSV：p4_pso_summary.csv / _plan.csv / _intervals.csv")
