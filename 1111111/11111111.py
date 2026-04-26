# -*- coding: utf-8 -*-
"""
第三问（FY1 投 3 枚）——分轮次的“单块优化”版本
每一轮只优化一个变量块；下一轮把上一轮的结果当作先验继续搜。
Round1: 优化 v, theta, (t1, tau1)
Round2: 仅优化 (t2, tau2) ，带约束 t2 >= t1 + 1
Round3: 仅优化 (t3, tau3) ，带约束 t3 >= t2 + 1
优化器：GA 粗搜 + SA 精炼（每轮都独立运行一次）

运行：python solve_q3_seq.py
"""

import math, time, os
import numpy as np

# =========================
# 0) 物理与几何（与前面保持一致）
# =========================
g = 9.8
R_cloud = 10.0
smoke_duration = 20.0
sink_v = 3.0
vm = 300.0
EVAL_DT = 0.02

T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
u_m = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)
T_HIT = np.linalg.norm(M0) / vm

FY1_F0 = np.array([17800.0, 0.0, 1800.0])

def missile_pos_vec(t):
    t = np.asarray(t, dtype=float)
    return M0[None, :] + (vm * t[:, None]) * u_m[None, :]

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9: out[-1] = (la, max(lb, b))
        else: out.append((a, b))
    return out

def total_length(intervals): return sum(b - a for a, b in intervals)

def explosion_point(F0, v, theta_rad, t_drop, tau):
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)

def occlusion_signal_point(E, t_e, dt=EVAL_DT):
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0: return []
    Tn = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, Tn)
    tabs = t_e + tgrid
    M = missile_pos_vec(tabs)
    C = E[None, :] + np.c_[np.zeros_like(tgrid), np.zeros_like(tgrid), -sink_v * tgrid]
    S = T_POINT[None, :] - M
    L2 = np.einsum('ij,ij->i', S, S)
    L2 = np.where(L2 > 1e-12, L2, 1e-12)
    CM = C - M
    s = np.clip(np.einsum('ij,ij->i', CM, S) / L2, 0.0, 1.0)
    Q = M + s[:, None] * S
    d = np.linalg.norm(C - Q, axis=1)
    inside = d <= R_cloud
    if not np.any(inside): return []
    intervals = []
    i = 0
    while i < Tn - 1:
        if inside[i]:
            j = i + 1
            while j < Tn and inside[j]: j += 1
            t_in = tgrid[i]
            if i > 0 and not inside[i - 1]:
                f1, f2 = d[i - 1] - R_cloud, d[i] - R_cloud
                t_in = tgrid[i - 1] + (tgrid[i] - tgrid[i - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            t_out = tgrid[j - 1]
            if j < Tn and not inside[j]:
                f1, f2 = d[j - 1] - R_cloud, d[j] - R_cloud
                t_out = tgrid[j - 1] + (tgrid[j] - tgrid[j - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            intervals.append((t_e + t_in, t_e + t_out))
            i = j
        else:
            i += 1
    return merge_intervals(intervals)

def verify_cover_time_point_target(uav_list, dt=EVAL_DT, print_details=True):
    all_intervals, per_bomb = [], []
    for uav in uav_list:
        name = uav.get("name", "FY1")
        F0 = np.array(uav.get("F0", FY1_F0), dtype=float)
        v = float(uav["v"])
        th = math.radians(float(uav["theta_deg"]))
        t_drop = uav["t_drop"]; tau = uav["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        for k, (td, ta) in enumerate(zip(t_drop, tau), start=1):
            R, E, t_e = explosion_point(F0, v, th, float(td), float(ta))
            if E[2] <= 0.0:
                if print_details:
                    print(f"[WARN] {name}#{k} 起爆高度 E_z={E[2]:.3f} <= 0，跳过该弹")
                continue
            ivs = occlusion_signal_point(E, t_e, dt=dt)
            all_intervals += ivs
            dur = total_length(ivs)
            per_bomb.append((name, k, ivs[0][0], ivs[-1][1], dur, E[2], t_e) if ivs
                            else (name, k, np.nan, np.nan, 0.0, E[2], t_e))
    union_ivs = merge_intervals(all_intervals)
    total = total_length(union_ivs)
    if print_details:
        print("\n=== 验证结果（目标当作点） ===")
        print(f"- 并集遮蔽总时长: {total:.6f} s")
        for i, (a, b) in enumerate(union_ivs, 1):
            print(f"  · 并集区间{i}: [{a:.6f}, {b:.6f}] s，时长 {(b-a):.6f} s")
        for (name, k, tin, tout, dur, Ez, te) in per_bomb:
            print(f"- {name}#{k}: t_e={te:.6f}, E_z={Ez:.3f}, 自身遮蔽={dur:.6f}"
                  + ("" if np.isnan(tin) else f"，首段≈[{tin:.6f},{tout:.6f}]"))
    return total, union_ivs, per_bomb

# =========================
# 1) 计划与“状态”（先验）
# =========================
class PlanState:
    """顺序累积的先验：Round1 确定 v,theta 和第一枚；Round2 加第二枚；Round3 加第三枚。"""
    __slots__ = ("v","theta_deg","t1","tau1","t2","tau2","t3","tau3")
    def __init__(self):
        self.v = None; self.theta_deg = None
        self.t1 = None; self.tau1 = None
        self.t2 = None; self.tau2 = None
        self.t3 = None; self.tau3 = None

    def fixed_uav_dict(self):
        """把已定的弹转为核算输入。"""
        drops, taus = [], []
        for t, z in [(self.t1,self.tau1),(self.t2,self.tau2),(self.t3,self.tau3)]:
            if (t is not None) and (z is not None):
                drops.append(float(t)); taus.append(float(z))
        return dict(name="FY1", F0=FY1_F0, v=float(self.v), theta_deg=float(self.theta_deg),
                    t_drop=drops, tau=taus)

# =========================
# 2) 每轮目标函数（只优当轮那一块）
# =========================
def obj_round1(x):
    """x = [v, theta_deg, t1, tau1]"""
    v, th, t1, z1 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    # 边界
    if not (70.0 <= v <= 140.0): return 1e9
    th = ((th + 180.0) % 360.0) - 180.0
    t1 = np.clip(t1, 0.0, max(0.0, T_HIT - 2.0))
    z1 = np.clip(z1, 0.0, 15.0)
    # 地下起爆罚
    _, E, _ = explosion_point(FY1_F0, v, math.radians(th), t1, z1)
    penalty = 5e5 if E[2] <= 0.0 else 0.0
    total, _, _ = verify_cover_time_point_target([dict(name="FY1", F0=FY1_F0, v=v, theta_deg=th,
                                                       t_drop=[t1], tau=[z1])],
                                                 dt=EVAL_DT, print_details=False)
    return -float(total) + penalty

def obj_round2(x, state: PlanState):
    """x = [t2, tau2]，约束：t2 >= t1 + 1"""
    t2, z2 = float(x[0]), float(x[1])
    if state.t1 is None or state.tau1 is None: return 1e9
    t_low = state.t1 + 1.0
    t2 = np.clip(t2, t_low, T_HIT)
    z2 = np.clip(z2, 0.0, 15.0)
    _, E, _ = explosion_point(FY1_F0, state.v, math.radians(state.theta_deg), t2, z2)
    penalty = 5e5 if E[2] <= 0.0 else 0.0
    uav = dict(name="FY1", F0=FY1_F0, v=state.v, theta_deg=state.theta_deg,
               t_drop=[state.t1, t2], tau=[state.tau1, z2])
    total, _, _ = verify_cover_time_point_target([uav], dt=EVAL_DT, print_details=False)
    # 软约束（安全起见再罚一次）
    gap_pen = 0.0 if (t2 - state.t1 >= 1.0) else 1e6
    return -float(total) + penalty + gap_pen

def obj_round3(x, state: PlanState):
    """x = [t3, tau3]，约束：t3 >= t2 + 1"""
    t3, z3 = float(x[0]), float(x[1])
    if state.t2 is None or state.tau2 is None: return 1e9
    t_low = state.t2 + 1.0
    t3 = np.clip(t3, t_low, T_HIT)
    z3 = np.clip(z3, 0.0, 15.0)
    _, E, _ = explosion_point(FY1_F0, state.v, math.radians(state.theta_deg), t3, z3)
    penalty = 5e5 if E[2] <= 0.0 else 0.0
    uav = dict(name="FY1", F0=FY1_F0, v=state.v, theta_deg=state.theta_deg,
               t_drop=[state.t1, state.t2, t3], tau=[state.tau1, state.tau2, z3])
    total, _, _ = verify_cover_time_point_target([uav], dt=EVAL_DT, print_details=False)
    gap_pen = 0.0 if (t3 - state.t2 >= 1.0) else 1e6
    return -float(total) + penalty + gap_pen

# =========================
# 3) 单轮 GA + SA（只优化当前块）
# =========================
def ga_sa_minimize(objective, x_init=None, bounds=None, seed=0,
                   pop_size=60, generations=80, mut_rate=0.35,
                   iters_sa=2000, T0=0.6, Tmin=1e-3):
    rng = np.random.default_rng(seed)

    def sample_uniform(lo, hi): return lo + (hi - lo) * rng.random()
    def clamp(x):
        if bounds is None: return x
        y = np.array(x, dtype=float)
        for i,(lo,hi) in enumerate(bounds): y[i] = min(max(y[i], lo), hi)
        return y

    # 初始化：若有初值，在其附近撒点；否则均匀采样
    pop = []
    for _ in range(pop_size):
        if x_init is None:
            xi = [sample_uniform(lo, hi) for (lo,hi) in bounds]
        else:
            xi = []
            for i,(lo,hi) in enumerate(bounds):
                span = hi - lo
                xi.append(float(np.clip(x_init[i] + rng.normal(0, 0.2*span), lo, hi)))
        pop.append(np.array(xi, dtype=float))
    fit = np.array([objective(x) for x in pop], dtype=float)

    # GA
    def tournament(k=3):
        idx = rng.integers(0, len(pop), size=k)
        j = idx[np.argmin(fit[idx])]
        return pop[j]

    def blx(a,b,alpha=0.3):
        lo = np.minimum(a,b); hi = np.maximum(a,b)
        span = hi - lo
        return lo - alpha*span + (1+2*alpha)*span*rng.random(size=a.shape)

    elite_k = max(2, len(pop)//15)
    for _ in range(generations):
        elite_idx = np.argsort(fit)[:elite_k]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            child = blx(p1, p2)
            # 变异
            if rng.random() < mut_rate:
                for i,(lo,hi) in enumerate(bounds):
                    span = hi - lo
                    child[i] += rng.normal(0, 0.1*span)
            new_pop.append(clamp(child))
        pop = new_pop
        fit = np.array([objective(x) for x in pop], dtype=float)

    gbest = pop[int(np.argmin(fit))]
    gbest_f = float(np.min(fit))

    # SA，从 GA 最优出发
    cur = gbest.copy(); curE = gbest_f
    best = cur.copy(); bestE = curE
    for it in range(1, iters_sa+1):
        T = max(Tmin, T0 * (0.995 ** it))
        # 自适应扰动
        step = cur.copy()
        for i,(lo,hi) in enumerate(bounds):
            span = hi - lo
            step[i] += rng.normal(0, 0.15*span*(0.2 + 0.8*(1 - it/iters_sa)))
        step = clamp(step)
        E = objective(step)
        dE = E - curE
        if dE <= 0 or rng.random() < math.exp(-dE / max(1e-12, T)):
            cur, curE = step, E
            if curE < bestE:
                best, bestE = cur.copy(), curE
    return best, bestE

# =========================
# 4) 顺序“一次只搜一个”接口
# =========================
def optimize_round1(state: PlanState, seed=42):
    # 决策变量： [v, theta_deg, t1, tau1]
    bounds = [(70.0,140.0), (-180.0,180.0), (0.0, max(0.0,T_HIT-2.0)), (0.0,15.0)]
    # 初值（如果 state 有先验就用，否则为空）
    x0 = None
    if state.v is not None:
        x0 = [state.v, state.theta_deg,
              state.t1 if state.t1 is not None else 0.2*T_HIT,
              state.tau1 if state.tau1 is not None else 4.0]
    t0 = time.time()
    x_best, f_best = ga_sa_minimize(obj_round1, x_init=x0, bounds=bounds, seed=seed)
    t1 = time.time()
    v, th, t1d, z1 = float(x_best[0]), float(x_best[1]), float(x_best[2]), float(x_best[3])
    state.v, state.theta_deg, state.t1, state.tau1 = v, ((th+180)%360)-180, t1d, z1
    print(f"[Round1] best obj={f_best:.6f} (cover≈{-f_best:.6f}s if无罚) | time={t1-t0:.2f}s")
    return state

def optimize_round2(state: PlanState, seed=7):
    if state.t1 is None: raise ValueError("Round2 需要 Round1 的先验（已确定 t1,tau1,v,theta）")
    # 决策变量：[t2, tau2]，t2 下界依赖 t1
    t2_lo = state.t1 + 1.0
    bounds = [(t2_lo, T_HIT), (0.0, 15.0)]
    x0 = [min(t2_lo + 0.1*(T_HIT-t2_lo), T_HIT), 4.0]
    t0 = time.time()
    obj = lambda x: obj_round2(x, state)
    x_best, f_best = ga_sa_minimize(obj, x_init=x0, bounds=bounds, seed=seed)
    t1 = time.time()
    state.t2, state.tau2 = float(x_best[0]), float(x_best[1])
    print(f"[Round2] best obj={f_best:.6f} (marginal cover≈{-f_best:.6f}s if无罚) | time={t1-t0:.2f}s")
    return state

def optimize_round3(state: PlanState, seed=11):
    if state.t2 is None: raise ValueError("Round3 需要 Round2 的先验（已确定 t2,tau2）")
    t3_lo = state.t2 + 1.0
    bounds = [(t3_lo, T_HIT), (0.0, 15.0)]
    x0 = [min(t3_lo + 0.1*(T_HIT-t3_lo), T_HIT), 4.0]
    t0 = time.time()
    obj = lambda x: obj_round3(x, state)
    x_best, f_best = ga_sa_minimize(obj, x_init=x0, bounds=bounds, seed=seed)
    t1 = time.time()
    state.t3, state.tau3 = float(x_best[0]), float(x_best[1])
    print(f"[Round3] best obj={f_best:.6f} (marginal cover≈{-f_best:.6f}s if无罚) | time={t1-t0:.2f}s")
    return state

# =========================
# 5) Demo：按“只搜一个”的顺序跑三轮
# =========================
def main():
    state = PlanState()

    print("\n>>> Round 1：优化 v, theta, (t1, tau1)")
    state = optimize_round1(state, seed=42)

    print("\n>>> Round 2：在 Round1 先验上，只优化 (t2, tau2)")
    state = optimize_round2(state, seed=7)

    print("\n>>> Round 3：在前两枚先验上，只优化 (t3, tau3)")
    state = optimize_round3(state, seed=11)

    # 汇总评估（按题面打印风格）
    uav = state.fixed_uav_dict()
    print("\n>>> 最终三枚弹联合评估：")
    total, union_ivs, per_bomb = verify_cover_time_point_target([uav], dt=EVAL_DT, print_details=True)
    print(f"\n[最终遮蔽总时长] = {total:.6f} s")
    print(f"[方案] v={state.v:.6f} m/s, theta={state.theta_deg:.6f} deg; "
          f"t=[{state.t1:.3f},{state.t2:.3f},{state.t3:.3f}], "
          f"tau=[{state.tau1:.3f},{state.tau2:.3f},{state.tau3:.3f}]")

if __name__ == "__main__":
    main()
