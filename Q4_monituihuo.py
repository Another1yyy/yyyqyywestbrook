# -*- coding: utf-8 -*-
"""
第三问（FY1 投 3 枚）—— 粒子群算法（PSO）全局优化（单文件版）
- 物理核：抛体起爆 + 云团下沉 + 线段最近距离 + 并集合并
- 约束：投放时刻严格 >1s 间隔；起爆高度 >0；角度归一
- 搜索：PSO（惯性退火 + 速度限幅）
- 评估：PSO阶段 dt_coarse，加速；终局 dt_fine 复评与导出
- 输出：按校验器风格打印 + result1.xlsx（或 CSV 降级）
"""

import math, time, os
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
MIN_GAP = 1.0           # 相邻投放严格 >1 s
EPS     = 1e-6

# 场景
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)   # 真目标点
M0      = np.array([20000.0, 0.0, 2000.0], dtype=float)  # 导弹初始
u_m     = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)
T_HIT   = np.linalg.norm(M0) / vm

FY1_F0  = np.array([17800.0, 0.0, 1800.0])  # FY1 初始（等高直线）

# tau 的物理上界（以高度 1800m 估计，略留余量）
TAU_PHYS_MAX = math.sqrt(2.0 * FY1_F0[2] / g) - 1e-3   # ~19.18s
TAU_MAX = min(15.0, TAU_PHYS_MAX)                      # 题面设定上限与物理上限取最小


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
def enforce_min_gap_sorted(t1, t2, t3, gap=MIN_GAP, eps=EPS):
    """保证严格 > gap 的相邻间隔；必要时回退到边界"""
    t1 = float(np.clip(t1, 0.0, max(0.0, T_HIT - 2*gap - 2*eps)))
    t2 = max(float(t2), t1 + gap + eps)
    t3 = max(float(t3), t2 + gap + eps)
    # 若越界，回退
    if t3 > T_HIT:
        t3 = T_HIT
        t2 = min(t2, t3 - gap - eps)
        t2 = max(t2, t1 + gap + eps)
        if t2 > T_HIT - gap - eps:
            t2 = T_HIT - gap - eps
        if t2 < 0: t2 = 0.0
        t1 = min(t1, t2 - gap - eps)
        t1 = max(t1, 0.0)
    return t1, t2, t3

def repair_vector(x):
    """
    x: [v, theta_deg, t1, t2, t3, tau1, tau2, tau3]
    作用：硬约束修复 + 边界裁剪 + 角度归一 + 地上起爆保障
    """
    v, th, t1, t2, t3, z1, z2, z3 = map(float, x)
    v  = float(np.clip(v, 70.0, 140.0))
    th = wrap_deg(th)

    # 排序 + 间隔修复（保持 (t_i, tau_i) 的配对关系 → 先修 times，再在下面计算 Ez 约束 tau）
    ts = np.array([t1, t2, t3], dtype=float)
    idx = np.argsort(ts)
    ts = ts[idx]
    taus = np.array([z1, z2, z3], dtype=float)[idx]

    t1, t2, t3 = enforce_min_gap_sorted(ts[0], ts[1], ts[2], gap=MIN_GAP, eps=EPS)

    # tau 上界：物理 + 题面
    taus = np.clip(taus, 0.0, TAU_MAX)

    # 起爆高度修复：Ez = z_drop - 0.5 g tau^2 > 0
    # 对等高 UAV，z_drop = FY1_F0[2]
    z_drop = FY1_F0[2]
    tau_up = math.sqrt(max(1e-12, 2.0 * z_drop / g)) - 1e-6
    taus = np.minimum(taus, tau_up)

    # 回到原顺序对应
    inv = np.argsort(idx)
    z1, z2, z3 = taus[inv[0]], taus[inv[1]], taus[inv[2]]

    return np.array([v, th, t1, t2, t3, z1, z2, z3], dtype=float)

def build_uav_from_x(x):
    v, th, t1, t2, t3, z1, z2, z3 = map(float, x)
    # 再做一次时间排序与间隔保障，确保落盘严格可行
    t1, t2, t3 = enforce_min_gap_sorted(t1, t2, t3, MIN_GAP, EPS)
    return dict(name="FY1", F0=FY1_F0, v=v, theta_deg=wrap_deg(th),
                t_drop=[t1, t2, t3], tau=[z1, z2, z3])


# =========================
# 3) 目标函数（最小化）
# =========================
def obj_neg_cover(x, dt=EVAL_DT_COARSE):
    """返回 -CoverTime（越小越好），内部会做 repair"""
    xr = repair_vector(x)
    uav = build_uav_from_x(xr)
    total, _, _ = verify_cover_time_point_target([uav], dt=dt, print_details=False)
    return -float(total), xr


# =========================
# 4) 粒子群算法（PSO）
# =========================
class PSO:
    def __init__(self, dim, bounds, pop_size=20, iters=200,
                 w_start=1.5874, w_end=1.099551, c1=1.6, c2=1.6,
                 v_max_scale=1, seed=0):
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
        self.rng = np.random.default_rng(seed)

        # 初始化
        X = []
        for _ in range(pop_size):
            xi = self.bounds[:,0] + (self.bounds[:,1]-self.bounds[:,0]) * self.rng.random(self.dim)
            # 特殊：theta 用 wrap；times 做一次软排序，后续 repair 会硬修
            xi[1] = wrap_deg(xi[1])
            X.append(xi)
        self.X = np.vstack(X)                           # (N,D)
        self.V = self.rng.normal(0, 0.1, size=(pop_size, dim)) * self.vmax  # 初始速度
        self.P = self.X.copy()                          # 个体历史最优
        self.Pf = np.full(pop_size, np.inf, dtype=float)
        self.G = None                                   # 全局最优位置
        self.Gf = np.inf
        self.Gx_repaired = None                         # repair 后的 G（便于落盘）

    def clamp(self, X):
        return np.clip(X, self.bounds[:,0], self.bounds[:,1])

    def step(self, objective, dt_eval):
        # 评估（带 repair）
        for i in range(self.pop_size):
            f, xr = objective(self.X[i], dt=dt_eval)
            if f < self.Pf[i]:
                self.Pf[i] = f
                self.P[i] = self.X[i].copy()
            if f < self.Gf:
                self.Gf = f
                self.G = self.X[i].copy()
                self.Gx_repaired = xr.copy()

        # 惯性线性退火
        # 也可用非线性退火（如指数），此处简洁处理
        # 在外层循环里动态计算 w
        pass

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
    v, th, t1, t2, t3, z1, z2, z3 = map(float, best_x_repaired)
    rows = []
    for (name, k, tin, tout, dur, Ez, te) in per_bomb:
        rows.append({
            "uav": name, "bomb_idx": k,
            "t_in": tin, "t_out": tout, "dur": dur,
            "E_z": Ez, "t_explode": te
        })
    meta = {
        "v_uav": v, "theta_deg": th,
        "t_drop_1": t1, "t_drop_2": t2, "t_drop_3": t3,
        "tau_1": z1, "tau_2": z2, "tau_3": z3,
        "total_cover_time": total_time
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

    # 变量：x=[v, theta_deg, t1, t2, t3, tau1, tau2, tau3]
    dim = 12
    bounds = [
        (70.0, 140.0),       # v1
        (-180.0, 180.0),     # theta_deg1
        (70.0, 140.0),  # v2
        (-180.0, 180.0),  # theta_deg2
        (70.0, 140.0),  # v3
        (-180.0, 180.0),  # theta_deg3
        (0.0, T_HIT),        # t1
        (0.0, T_HIT),        # t2
        (0.0, T_HIT),        # t3
        (0.0, TAU_MAX),      # tau1
        (0.0, TAU_MAX),      # tau2
        (0.0, TAU_MAX),      # tau3
    ]

    pso = PSO(dim, bounds, pop_size=150, iters=220,
              w_start=0.9, w_end=1, c1=1.7, c2=1.7,
              v_max_scale=1)

    # 运行 PSO（粗评估）
    best_xr, best_obj = pso.run(obj_neg_cover, dt_eval=EVAL_DT_COARSE, verbose=True)
    est_cover = -best_obj

    # 终局复评（细评估）
    uav = build_uav_from_x(best_xr)
    print("\n>>> 终局复评（细步长）与打印：")
    total_time, union_ivs, per_bomb = verify_cover_time_point_target([uav], dt=EVAL_DT_FINE, print_details=True)

    t1 = time.time()

    # 汇总打印
    v, th, t1d, t2d, t3d, z1, z2, z3 = map(float, best_xr)
    print("\n=== 最终方案（FY1，第三问，PSO） ===")
    print(f"v = {v:.6f} m/s, theta = {th:.6f} deg")
    print("t_drop =", [t1d, t2d, t3d])
    print("tau    =", [z1, z2, z3])
    print(f"[PSO估计] 遮蔽总时长（粗评）≈ {est_cover:.6f} s")
    print(f"[细评复核] 遮蔽总时长       = {total_time:.6f} s")
    print(f"[耗时统计] 总时间 = {t1 - t0:.3f} s")

    # 导出
    export_result(best_xr, total_time, union_ivs, per_bomb, xlsx_path="result1.xlsx")

if __name__ == "__main__":
    main()
