import math
import numpy as np

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

    def inside(t): return f(t) <= 0.0

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
    thetas = np.linspace(0, 2*np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0.0, height, n_z_side+1)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0] + radius*c,
                        base[1] + radius*s,
                        base[2] + z])

    # 上/下底面的极坐标环
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]  # 去掉 r=0，只取环
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0] + r*c,
                            base[1] + r*s,
                            base[2] + z])

    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    """
    单点 P 到一批线段 [M_i, X_i] 的距离（向量化）
    Ms, Xs: (N,3)
    返回: distances(N,), proj_s(N,)  (投影参数 ∈[0,1])
    """
    BA = Xs - Ms                # (N,3)
    BA2 = np.einsum('ij,ij->i', BA, BA)  # (N,)
    # 处理零长线段的稳健性
    zero = BA2 < 1e-12
    BA2[zero] = 1.0  # 避免除0，稍后用特别分支修正

    PA = P[None, :] - Ms        # (N,3)
    s = np.einsum('ij,ij->i', PA, BA) / BA2
    s = np.clip(s, 0.0, 1.0)
    Q = Ms + s[:, None] * BA    # (N,3)
    d = np.linalg.norm(P[None, :] - Q, axis=1)

    # 零长线段：距离退化为 |P - M|
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1)
        s[zero] = 0.0
    return d, s

# =========================
# 题目物理参数（第一问）
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 导弹（直指假目标原点）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

# FY1 无人机（朝 -x）
F0 = np.array([17800.0, 0.0, 1800.0])
v_u = 120.0
h = np.array([-1.0, 0.0, 0.0])
def drone_pos(t):
    return F0 + v_u * t * h

t_drop = 1.5
tau = 3.6
t_e = t_drop + tau

# 投放点与起爆点
R = drone_pos(t_drop)
E = R + v_u * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau**2)

def cloud_center(t):
    return E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)

# =========================
# 基于体积目标的遮蔽判定（ANY/ALL）
# =========================
def make_f_any_all():
    Xi = CYL_PTS  # (N,3)

    def f_any(t):
        # min_i dist(C(t), [M(t), Xi]) - R_cloud
        M = missile_pos(t)
        C = cloud_center(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)
        d, _ = dist_point_to_segments_batch(C, Ms, Xi)
        return float(np.min(d) - R_cloud)

    def f_all(t):
        # max_i dist(C(t), [M(t), Xi]) - R_cloud
        M = missile_pos(t)
        C = cloud_center(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)
        d, _ = dist_point_to_segments_batch(C, Ms, Xi)
        return float(np.max(d) - R_cloud)

    return f_any, f_all

# ========== 主流程 ==========
if __name__ == "__main__":
    f_any, f_all = make_f_any_all()

    # 选择口径：'ANY'（存在式）或 'ALL'（全遮蔽）
    MODE = 'ALL'   # 改为 'ALL' 可切换口径
    f = f_any if MODE == 'ANY' else f_all

    t0, t1 = t_e, t_e + effective_span
    intervals = find_cover_intervals(f, t0, t1, dt=0.02)
    total = sum(b - a for a, b in intervals)

    print("=== 第一问（体积目标，{} 口径）===".format(MODE))
    print(f"起爆时刻 t_e = {t_e:.6f} s；评估窗口 [{t0:.3f}, {t1:.3f}] s")
    print(f"投放点 R = ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f}) m")
    print(f"起爆点 E = ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m  (E_z>0?)")

    if intervals:
        for i, (a, b) in enumerate(intervals, 1):
            # 诊断：在区间内再采样寻找最小距离与其发生时刻（便于 sanity check）
            ts = np.linspace(a, b, 1501)
            mins = []
            for tt in ts:
                M = missile_pos(tt)
                C = cloud_center(tt)
                Ms = np.repeat(M[None, :], len(CYL_PTS), axis=0)
                d, _ = dist_point_to_segments_batch(C, Ms, CYL_PTS)
                val = (np.min(d) if MODE=='ANY' else np.max(d)) - R_cloud
                mins.append(val)
            k = int(np.argmin(mins))
            extreme = (np.min(mins) if MODE=='ANY' else np.min(mins))  # 数值最小即“最靠近/最紧”的情况
            print(f"- 区间{i}: 进入 {a:.6f} s, 离开 {b:.6f} s, 时长 {b-a:.6f} s; "
                  f"{'min' if MODE=='ANY' else 'max'}(D-R)≈{mins[k]:.4f} @ t≈{ts[k]:.6f}s")
        print(f"===> 有效遮蔽总时长: {total:.6f} s")
    else:
        print("未发生遮蔽。")

    # 备注：如需更严密，可把 CYL_PTS 采样密度调高；或把 dt 调小（更精, 更慢）。
