#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题 2 完整代码 —— 粒子群算法（PSO）版
"""
import math
import numpy as np

# ==========================================================
# 通用工具
# ==========================================================
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
        mid = 0.5 * (lo + hi); fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol: return mid
        if fa * fm <= 0: hi, fb = mid, fm
        else: lo, fa = mid, fm
    return 0.5 * (lo + hi)

def find_cover_intervals(f, t0, t1, dt=0.03):
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
    intervals, cur_in, cursor = [], inside(t0), t0
    for r in roots:
        if cur_in:
            intervals.append((cursor, r)); cur_in = False
        else:
            cursor, cur_in = r, True
    if cur_in: intervals.append((cursor, t1))
    return [(a, b) for a, b in intervals if b > a + 1e-8]

# ==========================================================
# 圆柱体目标
# ==========================================================
TARGET_BASE = np.array([0.0, 200.0, 0.0])
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
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0]+r*c, base[1]+r*s, base[2]+z])
    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    BA = Xs - Ms
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

# ==========================================================
# 题面参数
# ==========================================================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

F0 = np.array([17800.0, 0.0, 1800.0])
v_min, v_max = 70.0, 140.0

def cloud_center_builder(E, t_e):
    def C(t): return E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)
    return C

def build_volume_cover_fn(C_pts, missile_pos_func, cloud_center_func, mode='ANY'):
    Xi = C_pts
    if mode.upper() == 'ANY':
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None, :], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.min(d) - R_cloud)
        return f
    else:  # 'ALL'
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None, :], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

def drone_pos(t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau**2)
    return R, E

def evaluate_cover_time(v_u, theta, t_drop, tau, mode='ANY', scan_dt=0.03):
    if not (v_min <= v_u <= v_max): return -1.0, [], None
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return -1.0, [], None
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    intervals = find_cover_intervals(f, t_e, t_e + effective_span, dt=scan_dt)
    total = sum(b-a for a, b in intervals)
    info = dict(R=drone_pos(t_drop, v_u, theta), E=E, t_e=t_e)
    return total, intervals, info

def deg(theta): return (theta * 180.0 / math.pi) % 360.0

# ==========================================================
# 粒子群优化器
# ==========================================================
class PSO:
    def __init__(self, pop=80, max_iter=120, w=0.9, c1=2.0, c2=2.0,
                 mode='ANY', seed=42):
        self.pop, self.max_iter = pop, max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.lb = np.array([70.0, 0.0, 0.0, 1.0])
        self.ub = np.array([140.0, 2*np.pi, 60.0, 10.0])
        self.dim = 4

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _fitness(self, x):
        total, _, _ = evaluate_cover_time(
            x[0], x[1], x[2], x[3], mode=self.mode, scan_dt=0.03
        )
        return -max(total, 0.0)

    def run(self):
        X  = self.rng.uniform(self.lb, self.ub, (self.pop, self.dim))
        V  = self.rng.uniform(-1, 1, (self.pop, self.dim)) * (self.ub - self.lb) * 0.1
        pbest_X = X.copy()
        pbest_F = np.array([self._fitness(x) for x in X])
        gbest_idx = np.argmin(pbest_F)
        gbest_X, gbest_F = pbest_X[gbest_idx].copy(), pbest_F[gbest_idx]

        for it in range(self.max_iter):
            r1, r2 = self.rng.random((2, self.pop, self.dim))
            V = (self.w * V +
                 self.c1 * r1 * (pbest_X - X) +
                 self.c2 * r2 * (gbest_X - X))
            X = self._clip(X + V)
            for i in range(self.pop):
                f = self._fitness(X[i])
                if f < pbest_F[i]:
                    pbest_F[i], pbest_X[i] = f, X[i].copy()
                    if f < gbest_F:
                        gbest_F, gbest_X = f, X[i].copy()
            if it % 20 == 0 or it == self.max_iter - 1:
                print(f"[PSO] iter={it:3d}  best fitness={-gbest_F:.6f}  "
                      f"x={gbest_X}")
        best_total, best_intervals, best_info = evaluate_cover_time(
            gbest_X[0], gbest_X[1], gbest_X[2], gbest_X[3],
            mode=self.mode, scan_dt=0.02
        )
        return best_total, gbest_X, best_intervals, best_info

# ==========================================================
# 主入口
# ==========================================================
if __name__ == "__main__":
    MODE = "ALL"          # 可改为 "ANY"
    print(f"\n=== 题2 体积目标（{MODE} 口径）—— 粒子群优化 ===")
    pso = PSO(pop=100, max_iter=150, mode=MODE, seed=42)
    total, xopt, intervals, info = pso.run()

    v_u, theta, t_d, tau = xopt
    R, E, t_e = info["R"], info["E"], info["t_e"]

    print("\n=== PSO 最优方案 ===")
    print(f"- 总遮蔽时长: {total:.6f} s")
    print(f"- FY1 速度 v: {v_u:.3f} m/s")
    print(f"- FY1 航向角 θ: {deg(theta):.3f}°")
    print(f"- 投放时刻 t_d: {t_d:.6f} s")
    print(f"- 引信延时  τ: {tau:.6f} s")
    print(f"- 投放点 R: ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f}) m")
    print(f"- 起爆点 E: ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m")
    print(f"- 起爆时刻 t_e: {t_e:.6f} s")

    if intervals:
        for k, (a, b) in enumerate(intervals, 1):
            print(f"  · 区间{k}: [{a:.6f}, {b:.6f}] s，时长 {b-a:.6f} s")
    else:
        print("（未形成有效遮蔽区间）")