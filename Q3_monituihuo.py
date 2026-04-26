# -*- coding: utf-8 -*-
"""
第三问（FY1 投 3 枚）：单文件解法
- 搜索：遗传算法（GA） + 模拟退火（SA）
- 物理与遮蔽判定：内嵌自校验器（点目标、云团下沉、线段最近距离、并集区间合并）
- 输出：完全复用校验器的打印样式；另外导出 result1.xlsx（或降级为 result1.csv）

作者：Westbrook 项目适配版
"""

import math, time, os
import numpy as np

# =========================
# 0) 全局常量（与校验器一致）
# =========================
# 目标当作“点”
import numpy as np
g = 9.8
R_cloud = 10.0          # 云团有效半径 (m)
smoke_duration = 20.0   # 云团寿命 (s)
sink_v = 3.0            # 云团向下沉降 (m/s)
vm = 300.0              # 导弹速度 (m/s)
EVAL_DT = 0.02          # 评估步长（越小越准、越慢）  :contentReference[oaicite:4]{index=4}

T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)  # 真目标点  :contentReference[oaicite:5]{index=5}

# 导弹 M1：从 M0 直飞假目标原点 (0,0,0)
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
u_m = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)  # 朝原点单位向量
T_HIT = np.linalg.norm(M0) / vm                           # 抵达原点用时  :contentReference[oaicite:6]{index=6}

# UAV 默认初始位置
FY1_F0 = np.array([17800.0, 0.0, 1800.0])                 # 仅第三问用 FY1  :contentReference[oaicite:7]{index=7}

# =========================
# 1) 工具函数（与校验器一致）
# =========================
def missile_pos_vec(t):
    """导弹在绝对时刻 t（标量或数组）的位矢（向量化）。"""
    t = np.asarray(t, dtype=float)
    return M0[None, :] + (vm * t[:, None]) * u_m[None, :]   # :contentReference[oaicite:8]{index=8}

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out                                              # :contentReference[oaicite:9]{index=9}

def total_length(intervals):
    return sum(b - a for a, b in intervals)                 # :contentReference[oaicite:10]{index=10}

def explosion_point(F0, v, theta_rad, t_drop, tau):
    """
    根据 (F0, v, θ, t_drop, τ) 计算：
    - 投放点 R
    - 起爆点 E
    - 起爆时刻 t_e = t_drop + tau
    备注：θ 为平面航向角，x 轴为 0，逆时针为正。
    """
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)                             # :contentReference[oaicite:11]{index=11}

def occlusion_signal_point(E, t_e, dt=EVAL_DT):
    """
    针对“点目标”构造遮蔽信号：
    计算云团中心 C(t) 与线段 [ M(t) → T_POINT ] 的最短距离 d(t) 是否 <= R_cloud，
    返回绝对时刻的遮蔽区间列表。
    """
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0: return []

    # 时间网格（相对起爆时刻）
    Tn = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, Tn)
    tabs = t_e + tgrid

    # 向量化：导弹位置、云团中心
    M = missile_pos_vec(tabs)  # (T,3)
    C = E[None, :] + np.c_[np.zeros_like(tgrid), np.zeros_like(tgrid), -sink_v * tgrid]  # (T,3)

    # 线段最近距离（投影到 M->T 的线段）
    S = T_POINT[None, :] - M                                  # (T,3)
    L2 = np.einsum('ij,ij->i', S, S)
    L2 = np.where(L2 > 1e-12, L2, 1e-12)                      # 避免除零
    CM = C - M
    s = np.clip(np.einsum('ij,ij->i', CM, S) / L2, 0.0, 1.0)  # 投影系数
    Q = M + s[:, None] * S
    d = np.linalg.norm(C - Q, axis=1)                         # (T,)

    inside = d <= R_cloud
    if not np.any(inside): return []

    # 线性插值过零边界 → 相对时间区间，再平移回绝对时间
    intervals = []
    i = 0
    while i < Tn - 1:
        if inside[i]:
            j = i + 1
            while j < Tn and inside[j]:
                j += 1
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
    return merge_intervals(intervals)                          # :contentReference[oaicite:12]{index=12}

def verify_cover_time_point_target(uav_list, dt=EVAL_DT, print_details=True):
    """
    按校验器风格聚合多弹多机的遮蔽并集时间与区间并打印详情。
    返回：(total_time, union_intervals, per_bomb_intervals)
    """
    default_F0 = {"FY1": FY1_F0}
    all_intervals = []
    per_bomb = []  # (uav_name, idx, tin, tout, dur, Ez, te)

    for uav in uav_list:
        name = uav.get("name", "FY1")
        F0 = np.array(uav.get("F0", default_F0.get(name, FY1_F0)), dtype=float)
        v = float(uav["v"])
        th = math.radians(float(uav["theta_deg"]))

        t_drop = uav["t_drop"]; tau = uav["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), "t_drop 与 tau 长度不一致"

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
        if union_ivs:
            for i, (a, b) in enumerate(union_ivs, 1):
                print(f"  · 并集区间{i}: [{a:.6f}, {b:.6f}] s，时长 {(b-a):.6f} s")
        else:
            print("  · 无遮蔽区间")
        for (name, k, tin, tout, dur, Ez, te) in per_bomb:
            print(f"- {name}#{k}: 起爆时刻 t_e={te:.6f} s, 起爆高度 E_z={Ez:.3f} m, 自身遮蔽时长={dur:.6f} s"
                  + ("" if np.isnan(tin) else f"，首段≈[{tin:.6f},{tout:.6f}]"))
    return total, union_ivs, per_bomb                           # :contentReference[oaicite:13]{index=13}

# =========================
# 2) 第三问优化变量与评估：仅 FY1 投 3 枚
# =========================
class PlanFY1:
    __slots__ = ("v", "theta_deg", "t_drop", "tau")
    def __init__(self, v, theta_deg, t_drop, tau):
        self.v = float(v)
        self.theta_deg = float(theta_deg)
        self.t_drop = np.asarray(t_drop, dtype=float)
        self.tau = np.asarray(tau, dtype=float)

def _clip_and_sort_plan(p: PlanFY1) -> PlanFY1:
    v = float(np.clip(p.v, 70.0, 140.0))
    th = float(((p.theta_deg + 180.0) % 360.0) - 180.0)
    t = np.clip(p.t_drop, 0.0, T_HIT)
    z = np.clip(p.tau, 0.0, 15.0)
    pairs = sorted([(float(t[i]), float(z[i])) for i in range(3)], key=lambda x: x[0])
    t_sorted = np.array([x[0] for x in pairs], dtype=float)
    z_sorted = np.array([x[1] for x in pairs], dtype=float)
    return PlanFY1(v, th, t_sorted, z_sorted)

def _random_plan(rng: np.random.Generator) -> PlanFY1:
    v = rng.uniform(70.0, 140.0)
    theta = rng.uniform(-180.0, 180.0)
    t0 = rng.uniform(0.0, max(1.0, 0.15 * T_HIT))
    t1 = t0 + rng.uniform(1.0, 0.25 * T_HIT)
    t2 = t1 + rng.uniform(1.0, 0.25 * T_HIT)
    tau = rng.uniform(0.0, 12.0, size=3)
    return _clip_and_sort_plan(PlanFY1(v, theta, [t0, t1, t2], tau))

def _mutate_plan(p: PlanFY1, rng: np.random.Generator, sigma=0.12) -> PlanFY1:
    v = p.v + rng.normal(0, (140 - 70) * sigma)
    th = p.theta_deg + rng.normal(0, 180.0 * sigma * 0.2)
    t = p.t_drop + rng.normal(0, 0.10 * T_HIT * sigma, size=3)
    z = p.tau + rng.normal(0, 2.0 * sigma, size=3)
    return _clip_and_sort_plan(PlanFY1(v, th, t, z))

def _crossover_blx(p1: PlanFY1, p2: PlanFY1, rng: np.random.Generator, alpha=0.3) -> PlanFY1:
    def blx(a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        span = hi - lo
        return rng.uniform(lo - alpha * span, hi + alpha * span)
    v = float(blx(np.array([p1.v]), np.array([p2.v]))[0])
    v1 = np.array([math.cos(math.radians(p1.theta_deg)), math.sin(math.radians(p1.theta_deg))])
    v2 = np.array([math.cos(math.radians(p2.theta_deg)), math.sin(math.radians(p2.theta_deg))])
    vmean = v1 + v2
    th = math.degrees(math.atan2(vmean[1], vmean[0])) + rng.normal(0, 3.0)
    t = blx(p1.t_drop, p2.t_drop)
    z = blx(p1.tau, p2.tau)
    return _clip_and_sort_plan(PlanFY1(v, th, t, z))

def _evaluate_plan(p: PlanFY1, penalty=True, dt=None) -> float:
    """返回目标值：-遮蔽总时长 + 罚项（越小越好）。"""
    plan = _clip_and_sort_plan(p)
    # 基本硬约束：相邻投放 ≥ 1 s
    gap_ok = (plan.t_drop[1] - plan.t_drop[0] >= 1.0) and (plan.t_drop[2] - plan.t_drop[1] >= 1.0)

    # 快速过滤“地下起爆”
    bad_explode = 0
    for i in range(3):
        _, E, _ = explosion_point(FY1_F0, plan.v, math.radians(plan.theta_deg),
                                  float(plan.t_drop[i]), float(plan.tau[i]))
        if E[2] <= 0.0:
            bad_explode += 1

    uav = dict(name="FY1", F0=FY1_F0, v=plan.v, theta_deg=plan.theta_deg,
               t_drop=[float(x) for x in plan.t_drop], tau=[float(x) for x in plan.tau])
    total_time, _, _ = verify_cover_time_point_target([uav],
                                                      dt=(EVAL_DT if dt is None else dt),
                                                      print_details=False)
    obj = -float(total_time)
    if penalty:
        if not gap_ok:
            obj += 1e6
        obj += bad_explode * 5e5
    return obj

# =========================
# 3) 遗传算法（GA）与模拟退火（SA）
# =========================
def solve_ga(seed=42, pop_size=80, generations=120, elite_k=4, mut_rate=0.35, dt=None):
    rng = np.random.default_rng(seed)
    pop = [_random_plan(rng) for _ in range(pop_size)]
    fit = np.array([_evaluate_plan(p, dt=dt) for p in pop], dtype=float)

    def tournament(k=3):
        idx = rng.integers(0, pop_size, size=k)
        j = idx[np.argmin(fit[idx])]
        return pop[j]

    best_hist = []
    for _ in range(generations):
        elite_idx = np.argsort(fit)[:elite_k]
        new_pop = [pop[i] for i in elite_idx]
        best_hist.append(float(np.min(fit)))
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            child = _crossover_blx(p1, p2, rng)
            if rng.random() < mut_rate:
                child = _mutate_plan(child, rng, sigma=0.12)
            new_pop.append(child)
        pop = new_pop
        fit = np.array([_evaluate_plan(p, dt=dt) for p in pop], dtype=float)
    b = pop[int(np.argmin(fit))]
    return b, -_evaluate_plan(b, penalty=False, dt=dt), best_hist

def solve_sa(seed=8, iters=4000, T0=0.8, Tmin=1e-3, start_plan=None, dt=None):
    rng = np.random.default_rng(seed)
    cur = _random_plan(rng) if start_plan is None else _clip_and_sort_plan(start_plan)
    curE = _evaluate_plan(cur, dt=dt)
    best, bestE = cur, curE
    for it in range(1, iters + 1):
        T = max(Tmin, T0 * (0.995 ** it))
        sigma = 0.25 * (0.2 + 0.8 * (1 - it / iters))
        nxt = _mutate_plan(cur, rng, sigma=sigma)
        nxtE = _evaluate_plan(nxt, dt=dt)
        dE = nxtE - curE
        if dE <= 0 or rng.random() < math.exp(-dE / max(1e-12, T)):
            cur, curE = nxt, nxtE
            if curE < bestE:
                best, bestE = cur, curE
    return best, -_evaluate_plan(best, penalty=False, dt=dt)

# =========================
# 4) 结果导出（Excel/CSV）
# =========================
def export_result(best_plan, total_time, union_ivs, per_bomb,
                  xlsx_path="result1.xlsx", csv_fallback="result1.csv"):
    rows = []
    for (name, k, tin, tout, dur, Ez, te) in per_bomb:
        rows.append({
            "uav": name, "bomb_idx": k,
            "t_in": tin, "t_out": tout, "dur": dur,
            "E_z": Ez, "t_explode": te
        })
    meta = {
        "v_uav": best_plan.v, "theta_deg": best_plan.theta_deg,
        "t_drop_1": best_plan.t_drop[0], "t_drop_2": best_plan.t_drop[1], "t_drop_3": best_plan.t_drop[2],
        "tau_1": best_plan.tau[0], "tau_2": best_plan.tau[1], "tau_3": best_plan.tau[2],
        "total_cover_time": total_time
    }
    try:
        import pandas as pd
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
            pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="summary")
            pd.DataFrame(rows).to_excel(w, index=False, sheet_name="per_bomb")
            if union_ivs:
                pd.DataFrame([{"t_start": a, "t_end": b, "dur": (b-a)} for (a, b) in union_ivs])\
                    .to_excel(w, index=False, sheet_name="union_intervals")
        print(f"[导出] 已写入 {xlsx_path}")
        return xlsx_path
    except Exception as e:
        print(f"[导出] 写入 xlsx 失败（{e}），降级写 CSV。")
        # 简单 CSV 备选
        try:
            import csv
            with open(csv_fallback, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(list(meta.keys())); w.writerow(list(meta.values()))
                w.writerow([])
                w.writerow(["uav","bomb_idx","t_in","t_out","dur","E_z","t_explode"])
                for r in rows:
                    w.writerow([r["uav"], r["bomb_idx"], r["t_in"], r["t_out"], r["dur"], r["E_z"], r["t_explode"]])
            print(f"[导出] 已写入 {csv_fallback}")
            return csv_fallback
        except Exception as e2:
            print(f"[导出] CSV 亦失败：{e2}")
            return None

# =========================
# 5) 主入口
# =========================
def main():
    t0 = time.time()
    dt = EVAL_DT

    # GA 粗搜 + SA 精炼
    ga_best, ga_cover, _ = solve_ga(pop_size=80, generations=120, dt=dt)
    sa_best, sa_cover = solve_sa(iters=4000, T0=0.8, Tmin=1e-3, start_plan=ga_best, dt=dt)

    best = sa_best if sa_cover >= ga_cover else ga_best
    cover = max(sa_cover, ga_cover)
    t1 = time.time()

    # 打印与复核（完全按校验器风格）
    print("\n=== 最终方案（FY1，第三问） ===")
    print(f"v = {best.v:.6f} m/s, theta = {best.theta_deg:.6f} deg")
    print("t_drop =", [float(x) for x in best.t_drop])
    print("tau    =", [float(x) for x in best.tau])
    print(f"[搜索估计] 遮蔽总时长 ≈ {cover:.6f} s")
    print(f"[耗时统计] 优化用时 = {t1 - t0:.3f} s")

    uav = dict(name="FY1", F0=FY1_F0, v=best.v, theta_deg=best.theta_deg,
               t_drop=[float(x) for x in best.t_drop], tau=[float(x) for x in best.tau])

    print("\n>>> 复核打印（校验器逻辑）：")
    total_time, union_ivs, per_bomb = verify_cover_time_point_target([uav], dt=dt, print_details=True)
    print(f"\n[复核结果] 遮蔽总时长 = {total_time:.6f} s（应与上面接近/一致）")

    # 导出（题面第三问要求 result1.xlsx）  :contentReference[oaicite:14]{index=14}
    export_result(best, total_time, union_ivs, per_bomb, xlsx_path="result1.xlsx")

if __name__ == "__main__":
    main()
