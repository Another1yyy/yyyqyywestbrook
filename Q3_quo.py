#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题 3 —— 粒子群算法（PSO）版
- 决策向量：x = [v, theta, t1, tau1, t2, tau2, t3, tau3]
- 目标函数：最大化三枚遮蔽时间的“并集时长”
- 体积目标（圆柱体）遮蔽判定；ANY/ALL 两种口径可切换
- 仅依赖 numpy / math
"""
import math
import numpy as np

# ==========================================================
# 基础工具
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
    ivs = []
    cur_in, cursor = inside(t0), t0
    for r in roots:
        if cur_in:
            ivs.append((cursor, r)); cur_in = False
        else:
            cursor, cur_in = r, True
    if cur_in: ivs.append((cursor, t1))
    return [(a,b) for a,b in ivs if b > a + 1e-8]

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a,b))
    return out

def total_length(intervals):
    return sum(b-a for a,b in intervals)

def deg(rad): return (rad * 180.0 / math.pi) % 360.0

# ==========================================================
# 圆柱体目标采样（体积判定）
# ==========================================================
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

# ==========================================================
# 题面物理参数（导弹/无人机/云团）
# ==========================================================
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

# 变量约束
V_B   = (70.0, 140.0)        # 速度
TH_B  = (0.0, 2.0*math.pi)   # 航向
TD_B  = (0.0, 60.0)          # 投放时刻
TAU_B = (1.0, 10.0)          # 延时（与第2问脚本一致）
MIN_GAP = 1.0                # 同一无人机相邻投放间隔

# =========================
# 轨迹与遮蔽函数（体积目标）
# =========================
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

def build_volume_cover_fn(C_pts, missile_pos_func, cloud_center_func, mode='ANY'):
    Xi = C_pts
    if mode.upper() == 'ANY':
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.min(d) - R_cloud)
        return f
    else:  # ALL
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

def evaluate_single(v_u, theta, t_drop, tau, mode='ANY', dt=0.06):
    # 起爆高度检查
    R, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0:
        return [], dict(R=R, E=E, t_e=t_drop+tau)
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    ivs = find_cover_intervals(f, t_e, t_e + effective_span, dt=dt)
    return ivs, dict(R=R, E=E, t_e=t_e)

# ==========================================================
# 三枚并集适应度（含时间修正：排序 + 最小间隔）
# ==========================================================
def _enforce_times(t1, t2, t3, tmin=TD_B[0], tmax=TD_B[1], gap=MIN_GAP):
    t = np.sort(np.array([t1, t2, t3], dtype=float))
    # 先顺推满足间隔
    t[1] = max(t[1], t[0] + gap)
    t[2] = max(t[2], t[1] + gap)
    # 若超上界，整体左移
    excess = t[2] - tmax
    if excess > 0:
        t -= excess
        # 保底到下界
        if t[0] < tmin:
            shift = tmin - t[0]
            t += shift
            # 若仍越界（极端情形），压缩为紧贴边界的等间隔三点
            if t[2] > tmax:
                if 2*gap >= (tmax - tmin):  # 理论不可行，略收窄gap
                    gap = 0.9 * (tmax - tmin) / 2.0
                t[0] = tmin
                t[1] = t[0] + gap
                t[2] = t[1] + gap
    # 最终裁剪
    t = np.clip(t, tmin, tmax)
    # 再次确保间隔
    t[1] = min(max(t[1], t[0] + gap), tmax)
    t[2] = min(max(t[2], t[1] + gap), tmax)
    return float(t[0]), float(t[1]), float(t[2])

def evaluate_union_three(v_u, theta, t1, tau1, t2, tau2, t3, tau3,
                         mode='ANY', dt=0.06):
    # 强制排序 + 间隔
    t1, t2, t3 = _enforce_times(t1, t2, t3, gap=MIN_GAP)
    # 各枚区间
    ivs1, info1 = evaluate_single(v_u, theta, t1, tau1, mode=mode, dt=dt)
    ivs2, info2 = evaluate_single(v_u, theta, t2, tau2, mode=mode, dt=dt)
    ivs3, info3 = evaluate_single(v_u, theta, t3, tau3, mode=mode, dt=dt)
    # 并集
    merged = merge_intervals(ivs1 + ivs2 + ivs3)
    return total_length(merged), merged, (info1, info2, info3), (t1, t2, t3)

# ==========================================================
# 粒子群优化器（维度=8）
# ==========================================================
class PSO3:
    def __init__(self, pop=100, max_iter=160, w_start=0.8, w_end=0.4,
                 c1=1.6, c2=1.6, mode='ANY', seed=42):
        self.pop, self.max_iter = pop, max_iter
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        # 变量边界（与第2问风格保持一致：v放前、theta在第二位）
        self.lb = np.array([V_B[0], TH_B[0], TD_B[0], TAU_B[0],
                            TD_B[0], TAU_B[0], TD_B[0], TAU_B[0]], dtype=float)
        self.ub = np.array([V_B[1], TH_B[1], TD_B[1], TAU_B[1],
                            TD_B[1], TAU_B[1], TD_B[1], TAU_B[1]], dtype=float)
        self.dim = 8

        # 速度夹限，避免发散（经验值）
        self.vmin = np.array([-20.0, -math.pi, -6.0, -2.0, -6.0, -2.0, -6.0, -2.0])
        self.vmax = np.array([ 20.0,  math.pi,  6.0,  2.0,  6.0,  2.0,  6.0,  2.0])

    def _clip(self, X):
        Y = np.clip(X, self.lb, self.ub)
        # 角度 wrap
        Y[...,1] = np.mod(Y[...,1], 2.0*math.pi)
        return Y

    def _fitness(self, x):
        v, th, t1, tau1, t2, tau2, t3, tau3 = x
        total, _, _, _ = evaluate_union_three(
            v, th, t1, tau1, t2, tau2, t3, tau3,
            mode=self.mode, dt=0.06
        )
        # PSO 采用“最小化”——返回负的目标
        return -max(total, 0.0)

    def run(self):
        rng = self.rng
        # 初始化
        X = rng.uniform(self.lb, self.ub, size=(self.pop, self.dim))
        X = self._clip(X)
        V = rng.uniform(-1.0, 1.0, size=(self.pop, self.dim)) * (self.ub - self.lb) * 0.1

        pbest_X = X.copy()
        pbest_F = np.array([self._fitness(x) for x in X])
        gbest_idx = int(np.argmin(pbest_F))
        gbest_X, gbest_F = pbest_X[gbest_idx].copy(), float(pbest_F[gbest_idx])

        for it in range(1, self.max_iter+1):
            w = self.w_start + (self.w_end - self.w_start) * (it / self.max_iter)
            r1 = rng.random(size=X.shape)
            r2 = rng.random(size=X.shape)
            V = (w * V +
                 self.c1 * r1 * (pbest_X - X) +
                 self.c2 * r2 * (gbest_X - X))
            V = np.maximum(np.minimum(V, self.vmax), self.vmin)
            X = self._clip(X + V)

            # 更新个体/全局最优
            for i in range(self.pop):
                f = self._fitness(X[i])
                if f < pbest_F[i]:
                    pbest_F[i] = f
                    pbest_X[i] = X[i].copy()
                    if f < gbest_F:
                        gbest_F, gbest_X = f, X[i].copy()
            if it % 10 == 0 or it == 1 or it == self.max_iter:
                print(f"[PSO-3] iter {it:3d} | best(coarse) ≈ {-gbest_F:.4f} s  "
                      f"x*=[v={gbest_X[0]:.1f}, θ={deg(gbest_X[1]):.1f}°, t1={gbest_X[2]:.2f}, "
                      f"τ1={gbest_X[3]:.2f}, t2={gbest_X[4]:.2f}, τ2={gbest_X[5]:.2f}, "
                      f"t3={gbest_X[6]:.2f}, τ3={gbest_X[7]:.2f}]")

        # 高精复核
        v, th, t1, tau1, t2, tau2, t3, tau3 = gbest_X
        best_total, merged, infos, times = evaluate_union_three(
            v, th, t1, tau1, t2, tau2, t3, tau3, mode=self.mode, dt=0.03
        )
        return best_total, (v, th, *times, tau1, tau2, tau3), merged, infos

# ==========================================================
# 主入口
# ==========================================================
if __name__ == "__main__":
    MODE = "ALL"   # 可改 "ALL"
    print(f"\n=== 题3 体积目标（{MODE} 口径）—— 粒子群优化（3枚并集） ===")
    pso3 = PSO3(pop=500, max_iter=2500, mode=MODE, seed=42)
    total, xopt, union_iv, infos = pso3.run()

    v, th, t1, t2, t3, tau1, tau2, tau3 = xopt
    (info1, info2, info3) = infos
    R1, E1, te1 = info1["R"], info1["E"], info1["t_e"]
    R2, E2, te2 = info2["R"], info2["E"], info2["t_e"]
    R3, E3, te3 = info3["R"], info3["E"], info3["t_e"]

    print("\n=== PSO-3 最优方案（高精评估） ===")
    print(f"- 并集总时长: {total:.6f} s")
    print(f"- FY1 速度 v: {v:.3f} m/s")
    print(f"- FY1 航向 θ: {deg(th):.3f}°")
    print(f"- 投放 #1: t1={t1:.6f} s, τ1={tau1:.6f} s, t_e1={te1:.6f} s, "
          f"R1=({R1[0]:.3f},{R1[1]:.3f},{R1[2]:.3f}), "
          f"E1=({E1[0]:.3f},{E1[1]:.3f},{E1[2]:.3f})")
    print(f"- 投放 #2: t2={t2:.6f} s, τ2={tau2:.6f} s, t_e2={te2:.6f} s, "
          f"R2=({R2[0]:.3f},{R2[1]:.3f},{R2[2]:.3f}), "
          f"E2=({E2[0]:.3f},{E2[1]:.3f},{E2[2]:.3f})")
    print(f"- 投放 #3: t3={t3:.6f} s, τ3={tau3:.6f} s, t_e3={te3:.6f} s, "
          f"R3=({R3[0]:.3f},{R3[1]:.3f},{R3[2]:.3f}), "
          f"E3=({E3[0]:.3f},{E3[1]:.3f},{E3[2]:.3f})")

    if union_iv:
        for k,(a,b) in enumerate(union_iv, 1):
            print(f"  · 并集区间{k}: [{a:.6f}, {b:.6f}] s，时长 {b-a:.6f} s")
    else:
        print("（未形成遮蔽并集；可增大粒子数/迭代或减小 dt）")
