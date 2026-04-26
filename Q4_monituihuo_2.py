# -*- coding: utf-8 -*-
"""
第三问（3 架 UAV 各投 1 枚）—— 粒子群算法（PSO）全局优化（单文件版）
- 物理核：抛体起爆 + 云团下沉 + 线段最近距离 + 并集合并
- 约束：每架 UAV 仅 1 弹；起爆高度 > 0；角度归一
- 可选：全局“任意两弹间隔 > 1s”（默认关闭，可开启 ENFORCE_GLOBAL_GAP）
- 搜索：PSO（惯性退火 + 速度限幅）
- 评估：PSO阶段 dt_coarse，加速；终局 dt_fine 复评与导出
- 输出：按校验器风格打印 + result1.xlsx（或 CSV 降级）
"""

import math, time
import numpy as np

# =========================
# 0) 常量（题面/物理）
# =========================
g = 9.8
R_cloud = 10.0
smoke_duration = 20.0
sink_v = 3.0
vm = 300.0

# 评估步长
EVAL_DT_COARSE = 0.06   # PSO过程中（快）
EVAL_DT_FINE   = 0.02   # 收敛后复评/导出（准）

# 约束参数
MIN_GAP = 1.0           # “任意两弹间隔 > 1 s”的阈值（仅在 ENFORCE_GLOBAL_GAP=True 时生效）
EPS     = 1e-6
ENFORCE_GLOBAL_GAP = False   # 如需开启全局间隔硬约束，将其改为 True

# 场景
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)        # 真目标点
M0      = np.array([20000.0, 0.0, 2000.0], dtype=float)   # 导弹初始
u_m     = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)
T_HIT   = np.linalg.norm(M0) / vm

# 三架 UAV 的初始位置（等高直线）。⚠️如有题面给定，请在此处替换 FY2_F0 / FY3_F0
FY1_F0  = np.array([17800.0,    0.0, 1800.0])
FY2_F0  = np.array([12000.0, 1400, 1400.0])  # TODO: 若有明确给定，请替换
FY3_F0  = np.array([6000.0, -3000.0, 700.0])  # TODO: 若有明确给定，请替换

# 各机 tau 的物理上界（以各自高度估算；题面上限若更小取更小）
def tau_phys_max(z_drop):  # sqrt(2h/g)
    return math.sqrt(max(1e-12, 2.0 * z_drop / g)) - 1e-3
TAU_MAX_1 = min(15.0, tau_phys_max(FY1_F0[2]))
TAU_MAX_2 = min(15.0, tau_phys_max(FY2_F0[2]))
TAU_MAX_3 = min(15.0, tau_phys_max(FY3_F0[2]))


# =========================
# 1) 物理与几何工具
# =========================
def wrap_deg(theta_deg: float) -> float:
    """角度规约到 (-180, 180]"""
    return ((theta_deg + 180.0) % 360.0) - 180.0

def missile_pos_vec(t):
    """导弹在绝对时刻 t（标量或数组）的位矢。"""
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

def total_length(intervals):
    return sum(b - a for a, b in intervals)

def explosion_point(F0, v, theta_rad, t_drop, tau):
    """抛体：投放点 R、起爆点 E、起爆时刻 t_e"""
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)

def occlusion_signal_point(E, t_e, dt):
    """云心到视线段 [M(t)→T_POINT] 的最近距离 <= R_cloud → 遮蔽区间"""
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0: return []
    Tn = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, Tn)
    tabs = t_e + tgrid
    # 向量化
    M = missile_pos_vec(tabs)  # (T,3)
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
    # 边界线性插值
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

def verify_cover_time_point_target(uav_list, dt=EVAL_DT_FINE, print_details=True):
    """按‘校验器风格’聚合并打印"""
    all_intervals, per_bomb = [], []
    for uav in uav_list:
        name = uav.get("name", "FY?")
        F0 = np.array(uav["F0"], dtype=float)
        v  = float(uav["v"])
        th = math.radians(float(uav["theta_deg"]))
        t_drop = uav["t_drop"]; tau = uav["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        for k, (td, ta) in enumerate(zip(t_drop, tau), start=1):
            _, E, t_e = explosion_point(F0, v, th, float(td), float(ta))
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
    return total, union_ivs, per_bomb


# =========================
# 2) 约束修复（硬约束优先）
# =========================
def enforce_global_pairwise_gap(times, gap=MIN_GAP, eps=EPS):
    """
    对任意两弹的投放时刻施加全局最小间隔（严格 > gap）。
    策略：按时刻升序逐个前推；溢出 T_HIT 时整体左移，再做一次前推；确保位于 [0, T_HIT]。
    入参/出参均保持原 UAV 顺序。
    """
    t = np.array(times, dtype=float)
    idx = np.argsort(t)
    t_new = t.copy()

    # Forward push
    t_new[idx[0]] = np.clip(t_new[idx[0]], 0.0, T_HIT)
    for k in range(1, len(idx)):
        prev = t_new[idx[k-1]] + gap + eps
        t_new[idx[k]] = max(t_new[idx[k]], prev)

    # Overflow fix
    overflow = t_new[idx[-1]] - T_HIT
    if overflow > 0:
        t_new -= overflow
        t_new = np.clip(t_new, 0.0, T_HIT)
        # re-enforce forward gap once more
        t_new[idx[0]] = np.clip(t_new[idx[0]], 0.0, T_HIT)
        for k in range(1, len(idx)):
            prev = t_new[idx[k-1]] + gap + eps
            t_new[idx[k]] = max(t_new[idx[k]], prev)
        # if still overflow (极端情况), clamp last and shift left uniformly
        overflow2 = t_new[idx[-1]] - T_HIT
        if overflow2 > 0:
            t_new -= overflow2
            t_new = np.clip(t_new, 0.0, T_HIT)
    return t_new.tolist()

def repair_vector_3uav(x):
    """
    x: [v1, th1, t1, tau1,  v2, th2, t2, tau2,  v3, th3, t3, tau3]
    作用：硬约束修复 + 边界裁剪 + 角度归一 + 起爆高度保障 +（可选）全局最小间隔
    """
    v1, th1, t1, z1,  v2, th2, t2, z2,  v3, th3, t3, z3 = map(float, x)

    # 速度与角度
    v1 = float(np.clip(v1, 70.0, 140.0)); th1 = wrap_deg(th1)
    v2 = float(np.clip(v2, 70.0, 140.0)); th2 = wrap_deg(th2)
    v3 = float(np.clip(v3, 70.0, 140.0)); th3 = wrap_deg(th3)

    # 投放时刻裁剪
    t1 = float(np.clip(t1, 0.0, T_HIT))
    t2 = float(np.clip(t2, 0.0, T_HIT))
    t3 = float(np.clip(t3, 0.0, T_HIT))

    # tau 上界（题面与物理共同约束）
    z1 = float(np.clip(z1, 0.0, TAU_MAX_1))
    z2 = float(np.clip(z2, 0.0, TAU_MAX_2))
    z3 = float(np.clip(z3, 0.0, TAU_MAX_3))

    # 可选：全局任意两弹间隔 > MIN_GAP
    if ENFORCE_GLOBAL_GAP:
        t1, t2, t3 = enforce_global_pairwise_gap([t1, t2, t3], gap=MIN_GAP, eps=EPS)

    return np.array([v1, th1, t1, z1,  v2, th2, t2, z2,  v3, th3, t3, z3], dtype=float)

def build_uavs_from_x(x):
    v1, th1, t1, z1,  v2, th2, t2, z2,  v3, th3, t3, z3 = map(float, x)
    # 落盘前再做一次可选的全局间隔保障
    if ENFORCE_GLOBAL_GAP:
        t1, t2, t3 = enforce_global_pairwise_gap([t1, t2, t3], gap=MIN_GAP, eps=EPS)
    return [
        dict(name="FY1", F0=FY1_F0, v=v1, theta_deg=wrap_deg(th1), t_drop=[t1], tau=[z1]),
        dict(name="FY2", F0=FY2_F0, v=v2, theta_deg=wrap_deg(th2), t_drop=[t2], tau=[z2]),
        dict(name="FY3", F0=FY3_F0, v=v3, theta_deg=wrap_deg(th3), t_drop=[t3], tau=[z3]),
    ]


# =========================
# 3) 目标函数（最小化）
# =========================
def obj_neg_cover_3uav(x, dt=EVAL_DT_COARSE):
    """返回 -CoverTime（越小越好），内部会做 repair"""
    xr = repair_vector_3uav(x)
    uavs = build_uavs_from_x(xr)
    total, _, _ = verify_cover_time_point_target(uavs, dt=dt, print_details=False)
    return -float(total), xr


# =========================
# 4) 粒子群算法（PSO）
# =========================
class PSO:
    def __init__(self, dim, bounds, pop_size=120, iters=220,
                 w_start=0.90, w_end=0.35, c1=1.7, c2=1.7,
                 v_max_scale=0.25):
        """
        bounds: list[(lo,hi)] for each dim
        v_max_scale: 速度上限 = v_max_scale * (hi - lo)
        """
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.iters = iters
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.vmax = v_max_scale * (self.bounds[:,1] - self.bounds[:,0])
        self.rng = np.random.default_rng()

        # 初始化
        X = []
        for _ in range(pop_size):
            xi = self.bounds[:,0] + (self.bounds[:,1]-self.bounds[:,0]) * self.rng.random(self.dim)
            # 特殊：三个角度 wrap
            xi[1]  = wrap_deg(xi[1])
            xi[5]  = wrap_deg(xi[5])
            xi[9]  = wrap_deg(xi[9])
            X.append(xi)
        self.X = np.vstack(X)                           # (N,D)
        self.V = self.rng.normal(0, 0.1, size=(pop_size, dim)) * self.vmax  # 初始速度
        self.P = self.X.copy()                          # 个体历史最优
        self.Pf = np.full(pop_size, np.inf, dtype=float)
        self.G = None                                   # 全局最优位置
        self.Gf = np.inf
        self.Gx_repaired = None                         # repair 后的 G

    def clamp(self, X):
        return np.clip(X, self.bounds[:,0], self.bounds[:,1])

    def run(self, objective, dt_eval=EVAL_DT_COARSE, verbose=True):
        for it in range(1, self.iters + 1):
            # 评估 + Pbest/Gbest 更新
            for i in range(self.pop_size):
                f, xr = objective(self.X[i], dt=dt_eval)
                if f < self.Pf[i]:
                    self.Pf[i] = f
                    self.P[i] = self.X[i].copy()
                if f < self.Gf:
                    self.Gf = f
                    self.G = self.X[i].copy()
                    self.Gx_repaired = xr.copy()

            # 速度/位置更新
            w = self.w_start + (self.w_end - self.w_start) * (it / self.iters)
            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))
            cognitive = self.c1 * r1 * (self.P - self.X)
            social    = self.c2 * r2 * (self.G - self.X)
            self.V = w * self.V + cognitive + social
            # 速度限幅
            self.V = np.clip(self.V, -self.vmax, self.vmax)
            # 更新位置 + 边界裁剪
            self.X = self.X + self.V
            self.X = self.clamp(self.X)

            if verbose and (it % max(1, self.iters//10) == 0):
                print(f"[PSO] iter={it:4d}  best_obj={self.Gf:.6f}  est_cover={-self.Gf:.6f}s")

        return self.Gx_repaired, self.Gf


# =========================
# 5) 结果导出
# =========================
def export_result(best_x_repaired, total_time, union_ivs, per_bomb,
                  xlsx_path="result1.xlsx", csv_fallback="result1.csv"):
    v1, th1, t1, z1,  v2, th2, t2, z2,  v3, th3, t3, z3 = map(float, best_x_repaired)
    rows = []
    for (name, k, tin, tout, dur, Ez, te) in per_bomb:
        rows.append({
            "uav": name, "bomb_idx": k,
            "t_in": tin, "t_out": tout, "dur": dur,
            "E_z": Ez, "t_explode": te
        })
    meta = {
        "v1": v1, "theta1_deg": th1, "t1": t1, "tau1": z1,
        "v2": v2, "theta2_deg": th2, "t2": t2, "tau2": z2,
        "v3": v3, "theta3_deg": th3, "t3": t3, "tau3": z3,
        "total_cover_time": total_time,
        "enforce_global_gap": ENFORCE_GLOBAL_GAP
    }
    try:
        import pandas as pd
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
            pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="summary")
            pd.DataFrame(rows).to_excel(w, index=False, sheet_name="per_bomb")
            if union_ivs:
                pd.DataFrame([{"t_start": a, "t_end": b, "dur": (b-a)} for (a,b) in union_ivs])\
                    .to_excel(w, index=False, sheet_name="union_intervals")
        print(f"[导出] 已写入 {xlsx_path}")
        return xlsx_path
    except Exception as e:
        print(f"[导出] xlsx 失败（{e}），降级 CSV。")
        try:
            import csv
            with open(csv_fallback, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # Summary
                w.writerow(list(meta.keys())); w.writerow(list(meta.values()))
                w.writerow([])
                # Per-bomb
                w.writerow(["uav","bomb_idx","t_in","t_out","dur","E_z","t_explode"])
                for r in rows:
                    w.writerow([r["uav"], r["bomb_idx"], r["t_in"], r["t_out"], r["dur"], r["E_z"], r["t_explode"]])
            print(f"[导出] 已写入 {csv_fallback}")
            return csv_fallback
        except Exception as e2:
            print(f"[导出] CSV 亦失败：{e2}")
            return None


# =========================
# 6) 主入口
# =========================
def main():
    t0 = time.time()

    # 变量：x=[v1,th1,t1,tau1,  v2,th2,t2,tau2,  v3,th3,t3,tau3]
    dim = 12
    bounds = [
        (70.0, 140.0),  (-180.0, 180.0),  (0.0, T_HIT),  (0.0, TAU_MAX_1),  # UAV1
        (70.0, 140.0),  (-180.0, 180.0),  (0.0, T_HIT),  (0.0, TAU_MAX_2),  # UAV2
        (70.0, 140.0),  (-180.0, 180.0),  (0.0, T_HIT),  (0.0, TAU_MAX_3),  # UAV3
    ]

    # PSO 超参（可与此前强化调参器结合使用）
    pso = PSO(dim, bounds,
              pop_size=150, iters=220,
              w_start=0.9, w_end=1, c1=1.9, c2=1.9,
              v_max_scale=1.2)

    # 运行 PSO（粗评估）
    best_xr, best_obj = pso.run(obj_neg_cover_3uav, dt_eval=EVAL_DT_COARSE, verbose=True)
    est_cover = -best_obj

    # 终局复评（细评估）
    uavs = build_uavs_from_x(best_xr)
    print("\n>>> 终局复评（细步长）与打印：")
    total_time, union_ivs, per_bomb = verify_cover_time_point_target(uavs, dt=EVAL_DT_FINE, print_details=True)

    t1 = time.time()

    # 汇总打印
    v1, th1, t1d, z1,  v2, th2, t2d, z2,  v3, th3, t3d, z3 = map(float, best_xr)
    print("\n=== 最终方案（3 UAV，各 1 枚，PSO） ===")
    print(f"[ENFORCE_GLOBAL_GAP] = {ENFORCE_GLOBAL_GAP}")
    print(f"UAV1: v={v1:.3f} m/s, theta={th1:.3f} deg, t={t1d:.3f}, tau={z1:.3f}")
    print(f"UAV2: v={v2:.3f} m/s, theta={th2:.3f} deg, t={t2d:.3f}, tau={z2:.3f}")
    print(f"UAV3: v={v3:.3f} m/s, theta={th3:.3f} deg, t={t3d:.3f}, tau={z3:.3f}")
    print(f"[PSO估计] 遮蔽总时长（粗评）≈ {est_cover:.6f} s")
    print(f"[细评复核] 遮蔽总时长       = {total_time:.6f} s")
    print(f"[耗时统计] 总时间 = {t1 - t0:.3f} s")

    # 导出
    export_result(best_xr, total_time, union_ivs, per_bomb, xlsx_path="result1.xlsx")

if __name__ == "__main__":
    main()
