# -*- coding: utf-8 -*-
"""
Problem 2 via Bi-level Genetic Algorithm (outer: heading & speed; inner: drop time & fuse)
Target occlusion uses VOLUME (cylindrical) model with ANY/ALL modes.

Standalone: numpy + math only. Prints the best plan & intervals.
"""

import math
import numpy as np
from typing import Tuple, List

# =========================
# 基础与几何工具
# =========================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def bisect_root(f, a, b, tol=1e-10, maxiter=100):
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

def find_cover_intervals(f, t0, t1, dt=0.04):
    """
    在 [t0, t1] 扫描 f(t)=D(t)-R_cloud，返回所有满足 D<=R 的区间
    """
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t); vs.append(f(t)); t += dt

    roots = []
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        fa, fb = vs[i - 1], vs[i]
        if fa == 0.0: roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None: roots.append(r)
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
    return [(a, b) for a, b in intervals if b > a + 1e-8]

def total_length(intervals: List[Tuple[float,float]]) -> float:
    return sum(b - a for a, b in intervals)

# =========================
# 目标（体积）圆柱体采样（适度稀疏，便于 GA 速度）
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])   # 下底圆心
TARGET_R    = 7.0
TARGET_H    = 10.0

def sample_cylinder_points(base, radius, height,
                           n_theta_side=32, n_z_side=6,
                           n_theta_disk=24, n_r_disk=2):
    """
    适度稀疏的圆柱体表面采样（~320 点），GA 评估更快
    """
    pts = []

    # 侧面
    thetas = np.linspace(0, 2*np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0.0, height, n_z_side+1)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0] + radius*c, base[1] + radius*s, base[2] + z])

    # 上/下底
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]  # 去 r=0
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0] + r*c, base[1] + r*s, base[2] + z])

    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    """
    单点 P 到一批线段 [M_i, X_i] 的距离（向量化）
    Ms, Xs: (N,3) -> 返回 distances(N,), proj_s(N,)
    """
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
# 题面物理参数
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

# FY1 初始（等高）
F0 = np.array([17800.0, 0.0, 1800.0])
v_min, v_max = 70.0, 140.0

def drone_pos(t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5 * np.array([0.0,0.0,-g]) * (tau**2)
    return R, E

def cloud_center_builder(E, t_e):
    def C(t): return E + np.array([0.0,0.0,-sink_v]) * (t - t_e)
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
    else:  # 'ALL'
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

# 单次方案评估（返回时长与区间）
def evaluate_cover(v_u, theta, t_drop, tau, mode='ANY', scan_dt=0.04):
    if not (v_min <= v_u <= v_max): return -1.0, [], None
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return -1.0, [], None  # 起爆高度>0
    t_e = t_drop + tau
    C   = cloud_center_builder(E, t_e)
    f   = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    intervals = find_cover_intervals(f, t_e, t_e + effective_span, dt=scan_dt)
    total = total_length(intervals)
    info  = dict(R=drone_pos(t_drop, v_u, theta), E=E, t_e=t_e)
    return total, intervals, info

# =========================
# 遗传算法（实数编码）工具
# =========================
def tournament_select(fits, k=2, rng=None):
    n = len(fits); rng = np.random.default_rng() if rng is None else rng
    best_idx = None; best_val = -1e18
    for _ in range(k):
        i = rng.integers(0, n)
        if fits[i] > best_val:
            best_val = fits[i]; best_idx = i
    return best_idx

def blx_alpha_crossover(p1, p2, alpha, rng):
    """
    BLX-α 交叉：对子向量逐基因进行 [min-α·d, max+α·d] 采样
    """
    p1 = np.asarray(p1); p2 = np.asarray(p2)
    lo = np.minimum(p1, p2); hi = np.maximum(p1, p2)
    d  = hi - lo
    return rng.uniform(lo - alpha*d, hi + alpha*d)

def gaussian_mutation(x, sigma, bounds, rng):
    x = np.asarray(x).copy()
    x += rng.normal(0.0, sigma, size=x.shape)
    # 角度基因（第0位可能是角度）由调用方自行 wrap；这里仅做区间裁剪
    for i,(lo,hi) in enumerate(bounds):
        if lo is None or hi is None: continue
        x[i] = np.clip(x[i], lo, hi)
    return x

# =========================
# 内层 GA：优化 (t_drop, tau)  —— 给定 (theta, v)
# =========================
def inner_ga(theta, v_u, mode='ANY',
             bounds=((0.0, 60.0), (0.25, 9.0)),
             pop_size=20, gens=25,
             p_c=0.9, p_m=0.2, sigma=(1.0, 0.5),
             seed=0):
    rng = np.random.default_rng(seed)
    # 初始化
    pop = np.zeros((pop_size, 2), dtype=float)
    for i in range(pop_size):
        pop[i,0] = rng.uniform(bounds[0][0], bounds[0][1])  # t_drop
        pop[i,1] = rng.uniform(bounds[1][0], bounds[1][1])  # tau

    # 适应度（粗评 dt=0.08 提速）
    def fitness(ind):
        t_d, tau = ind
        total, _, _ = evaluate_cover(v_u, theta, t_d, tau, mode=mode, scan_dt=0.08)
        return total

    fits = np.array([fitness(ind) for ind in pop])
    best = (np.max(fits), pop[np.argmax(fits)].copy())

    for g in range(gens):
        new_pop = []
        while len(new_pop) < pop_size:
            i = tournament_select(fits, k=2, rng=rng)
            j = tournament_select(fits, k=2, rng=rng)
            p1 = pop[i]; p2 = pop[j]
            # 交叉
            if rng.random() < p_c:
                c = blx_alpha_crossover(p1, p2, alpha=0.5, rng=rng)
            else:
                c = p1.copy()
            # 变异
            if rng.random() < p_m:
                c = gaussian_mutation(c, sigma=np.array(sigma), bounds=bounds, rng=rng)
            # 投放/延时边界
            c[0] = float(np.clip(c[0], bounds[0][0], bounds[0][1]))
            c[1] = float(np.clip(c[1], bounds[1][0], bounds[1][1]))
            new_pop.append(c)
        pop = np.vstack(new_pop)
        fits = np.array([fitness(ind) for ind in pop])
        cur_best = (np.max(fits), pop[np.argmax(fits)].copy())
        if cur_best[0] > best[0]: best = cur_best

    # 对内层最优做一次细评（dt=0.03）
    t_d, tau = best[1]
    total, intervals, info = evaluate_cover(v_u, theta, t_d, tau, mode=mode, scan_dt=0.03)
    return total, float(t_d), float(tau), intervals, info

# =========================
# 外层 GA：优化 (theta, v)  —— 内层 GA 做适应度评估
# =========================
def outer_ga(mode='ANY',
             theta_bound=(0.0, 2.0*math.pi),
             v_bound=(70.0, 140.0),
             pop_size=18, gens=20,
             p_c=0.9, p_m=0.25, sigma=(8.0*math.pi/180.0, 6.0),
             seed=123):
    rng = np.random.default_rng(seed)

    # 初始化
    pop = np.zeros((pop_size, 2), dtype=float)
    for i in range(pop_size):
        pop[i,0] = rng.uniform(theta_bound[0], theta_bound[1])   # theta
        pop[i,1] = rng.uniform(v_bound[0],    v_bound[1])        # v

    # 适应度：调用内层 GA（粗略参数），减少总耗时
    def fitness(ind):
        th, v = ind
        total, t_d, tau, intervals, info = inner_ga(
            th, v, mode=mode,
            pop_size=16, gens=18,  # 内层略小
            seed=rng.integers(1, 1_000_000)
        )
        return total, (th, v, t_d, tau, intervals, info)

    fits, payloads = [], []
    for ind in pop:
        total, pack = fitness(ind)
        fits.append(total); payloads.append(pack)
    fits = np.array(fits)

    best_idx = int(np.argmax(fits))
    best_val, best_pack = fits[best_idx], payloads[best_idx]

    for g in range(gens):
        new_pop = []
        while len(new_pop) < pop_size:
            i = tournament_select(fits, k=2, rng=rng)
            j = tournament_select(fits, k=2, rng=rng)
            p1 = pop[i]; p2 = pop[j]
            # 交叉（BLX-α）
            if rng.random() < p_c:
                c = blx_alpha_crossover(p1, p2, alpha=0.5, rng=rng)
            else:
                c = p1.copy()
            # 变异（角度+速度）
            if rng.random() < p_m:
                c = c.copy()
                c[0] += rng.normal(0.0, sigma[0])        # theta
                c[1] += rng.normal(0.0, sigma[1])        # v
            # 变量边界与角度归一
            c[0] = float(c[0] % (2.0*math.pi))
            c[1] = float(np.clip(c[1], v_bound[0], v_bound[1]))
            new_pop.append(c)
        pop = np.vstack(new_pop)

        fits = []
        payloads = []
        for ind in pop:
            total, pack = fitness(ind)
            fits.append(total); payloads.append(pack)
        fits = np.array(fits)
        cur_idx = int(np.argmax(fits))
        cur_val, cur_pack = fits[cur_idx], payloads[cur_idx]
        if cur_val > best_val:
            best_val, best_pack = cur_val, cur_pack

        if (g+1) % 5 == 0:
            print(f"[Outer GA] gen {g+1:02d}: best union time (inner) ≈ {best_val:.4f} s")

    # 对外层最优解做一次“高精复核”（dt=0.02）
    th, v, t_d, tau, _, _ = best_pack
    final_total, intervals, info = evaluate_cover(v, th, t_d, tau, mode=mode, scan_dt=0.02)
    return final_total, (th, v, t_d, tau), intervals, info

# =========================
# 运行与输出
# =========================
def pretty_deg(rad): return (rad * 180.0 / math.pi) % 360.0

def main():
    MODE = "ALL"  # 可改为 "ALL"
    print(f"=== Problem 2 via Bi-level GA (mode={MODE}) ===")

    best_total, (theta, v_u, t_d, tau), intervals, info = outer_ga(
        mode=MODE,
        pop_size=18, gens=20,           # 外层规模/代数
        p_c=0.9, p_m=0.25,
        sigma=(8.0*math.pi/180.0, 6.0), # 角度/速度变异尺度
        seed=42
    )

    R = info["R"]; E = info["E"]; t_e = info["t_e"]

    print("\n=== 最优方案（Bi-level GA） ===")
    print(f"- 总遮蔽时长（高精复核 dt=0.02）: {best_total:.6f} s")
    print(f"- FY1 航向 θ: {pretty_deg(theta):.3f}°")
    print(f"- FY1 速度 v: {v_u:.3f} m/s")
    print(f"- 投放时刻 t_d: {t_d:.6f} s")
    print(f"- 引信延时  τ: {tau:.6f} s")
    print(f"- 起爆时刻 t_e: {t_e:.6f} s，评估窗口 [{t_e:.6f}, {t_e+20.0:.6f}] s")
    print(f"- 投放点 R: ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f}) m")
    print(f"- 起爆点 E: ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m  (E_z>0?)")

    if intervals:
        for k,(a,b) in enumerate(intervals,1):
            print(f"  · 区间{k}: [{a:.6f}, {b:.6f}] s，时长 {b-a:.6f} s")
    else:
        print("（未形成遮蔽区间，可加大 GA 规模或细化评估步长）")

if __name__ == "__main__":
    main()
