# -*- coding: utf-8 -*-
"""
问题 5 —— 5 架 UAV、每架至多 3 枚烟幕弹，对 3 枚导弹 (M1/M2/M3) 干扰
优化器：PSO（惯性退火 + 速度限幅），两阶段评估（粗/细）
目标：sum_{m∈{M1,M2,M3}} Cover_m  +  LAMBDA_INTERSECT * Cover_all_intersection
输出：result3.xlsx
"""

import math, time
import numpy as np

# =========================
# 0) 物理常量与场景
# =========================
g = 9.8
R_cloud = 10.0
smoke_duration = 20.0
sink_v = 3.0
vm = 300.0

# 评估步长
EVAL_DT_COARSE = 0.06   # PSO阶段（快）
EVAL_DT_FINE   = 0.02   # 复评导出（准）

# 目标加权：鼓励三导弹同时遮蔽
LAMBDA_INTERSECT = 0.25

# 约束
MIN_GAP = 1.0
EPS = 1e-6

# 坐标设置（题面给定）
# 目标：真目标下底面圆心 (0,200,0)；假目标原点 (0,0,0)
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)

# 导弹初始（朝原点飞）
M_INITS = {
    "M1": np.array([20000.0,     0.0, 2000.0]),
    "M2": np.array([19000.0,   600.0, 2100.0]),
    "M3": np.array([18000.0,  -600.0, 1900.0]),
}

def missile_dir_and_hit_time(M0):
    u = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)
    T_hit = np.linalg.norm(M0) / vm
    return u, T_hit

MISSILES = {}
T_HIT_MAX = 0.0
for name, M0 in M_INITS.items():
    u, th = missile_dir_and_hit_time(M0)
    MISSILES[name] = dict(M0=M0, u=u, T_hit=th, name=name)
    T_HIT_MAX = max(T_HIT_MAX, th)

# 五架 UAV 初始位置（题面给定）
FY0 = {
    "FY1": np.array([17800.0,     0.0, 1800.0]),
    "FY2": np.array([12000.0,  1400.0, 1400.0]),
    "FY3": np.array([ 6000.0, -3000.0,  700.0]),
    "FY4": np.array([11000.0,  2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}

def tau_phys_max(z_drop):  # sqrt(2h/g)
    return math.sqrt(max(1e-12, 2.0 * z_drop / g)) - 1e-3

TAU_MAXS = {
    k: min(15.0, tau_phys_max(v[2])) for k, v in FY0.items()
}

# =========================
# 1) 几何/评估工具
# =========================
def wrap_deg(theta_deg: float) -> float:
    """角度规约到 (-180, 180]"""
    return ((theta_deg + 180.0) % 360.0) - 180.0

def missile_pos_vec(t_abs, M0, u):
    """导弹在绝对时刻 t 的位矢（标量或数组）"""
    t = np.asarray(t_abs, dtype=float)
    return M0[None, :] + (vm * t[:, None]) * u[None, :]

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

def intersect_two(A, B):
    """两个并集区间列表的交集"""
    i = j = 0; out = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i]; b1, b2 = B[j]
        s, e = max(a1, b1), min(a2, b2)
        if s < e - 1e-12: out.append((s, e))
        if a2 < b2: i += 1
        else: j += 1
    return out

def intersect_many(list_of_lists):
    if not list_of_lists: return []
    cur = merge_intervals(list_of_lists[0])
    for k in range(1, len(list_of_lists)):
        cur = intersect_two(cur, merge_intervals(list_of_lists[k]))
        if not cur: break
    return cur

def explosion_point(F0, v, theta_rad, t_drop, tau):
    """抛体：投放点 R、起爆点 E、起爆时刻 t_e"""
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)

def occlusion_intervals_for_missile(E, t_e, missile, dt):
    """对指定导弹计算单枚云团的遮蔽区间"""
    span = min(smoke_duration, max(0.0, missile["T_hit"] - t_e))
    if span <= 0.0: return []
    Tn = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, Tn)
    tabs = t_e + tgrid
    M = missile_pos_vec(tabs, missile["M0"], missile["u"])
    C = E[None, :] + np.c_[np.zeros_like(tgrid), np.zeros_like(tgrid), -sink_v * tgrid]
    S = T_POINT[None, :] - M
    L2 = np.einsum('ij,ij->i', S, S); L2 = np.where(L2 > 1e-12, L2, 1e-12)
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

def evaluate_plan(uavs, dt=EVAL_DT_FINE, print_details=True):
    """
    输入 uavs: [ {name,F0,v,theta_deg,t_drop=[...],tau=[...]} × up to 5 ]
    输出：
      - per_missile_union: {M1: [(a,b),...], ...}
      - cover_sum: sum_m total_length(union_m)
      - cover_intersection: 三导弹并集的交集时长
      - per_bomb: 每枚弹在各导弹的自身遮蔽汇总
    """
    # 按导弹聚合
    per_missile_intervals = {m: [] for m in MISSILES.keys()}
    per_bomb_rows = []

    for uav in uavs:
        name = uav["name"]; F0 = np.array(uav["F0"], dtype=float)
        v = float(uav["v"]); th = math.radians(float(uav["theta_deg"]))
        t_drop = uav["t_drop"]; tau = uav["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        for k, (td, ta) in enumerate(zip(t_drop, tau), start=1):
            _, E, t_e = explosion_point(F0, v, th, float(td), float(ta))
            if E[2] <= 0.0:
                if print_details:
                    print(f"[WARN] {name}#{k} 起爆高度 E_z={E[2]:.3f} <= 0，跳过该弹")
                continue
            # 对每个导弹计算区间并累加
            for mname, miss in MISSILES.items():
                ivs = occlusion_intervals_for_missile(E, t_e, miss, dt)
                per_missile_intervals[mname] += ivs
                dur = total_length(ivs)
                per_bomb_rows.append(dict(
                    uav=name, bomb_idx=k, missile=mname,
                    t_e=t_e, E_z=float(E[2]), dur=dur,
                    t_first=ivs[0][0] if ivs else np.nan,
                    t_last=ivs[-1][1] if ivs else np.nan
                ))

    # 每个导弹取并集与时长
    per_missile_union = {m: merge_intervals(per_missile_intervals[m]) for m in MISSILES.keys()}
    per_missile_cover = {m: total_length(per_missile_union[m]) for m in MISSILES.keys()}
    cover_sum = sum(per_missile_cover.values())

    # 三导弹的“同时遮蔽”(交集)
    inter_list = [per_missile_union[m] for m in ["M1","M2","M3"]]
    inter_union = intersect_many(inter_list)
    cover_intersection = total_length(inter_union)

    if print_details:
        print("\n=== 多导弹评估（并集/交集） ===")
        for m in ["M1","M2","M3"]:
            print(f"- {m} 并集遮蔽: {per_missile_cover[m]:.6f} s, 区间数={len(per_missile_union[m])}")
        print(f"- 三导弹同时遮蔽(交集): {cover_intersection:.6f} s")

    return per_missile_union, per_missile_cover, cover_sum, inter_union, cover_intersection, per_bomb_rows

# =========================
# 2) 约束修复 / 变量映射
# =========================
UAV_NAMES = ["FY1","FY2","FY3","FY4","FY5"]
BOMBS_PER_UAV = 3

def enforce_min_gap_sorted(times, gap=MIN_GAP, eps=EPS, tmax=T_HIT_MAX):
    """保证严格 > gap；必要时回退。times 为升序/未升序均可。"""
    t = np.sort(np.array(times, dtype=float))
    if len(t) == 0: return t.tolist()
    t[0] = np.clip(t[0], 0.0, max(0.0, tmax - (len(t)-1)*(gap+eps)))
    for i in range(1, len(t)):
        lo = t[i-1] + gap + eps
        t[i] = max(t[i], lo)
    # 溢出则整体左移
    overflow = t[-1] - tmax
    if overflow > 0:
        t -= overflow
        t = np.clip(t, 0.0, tmax)
        for i in range(1, len(t)):
            lo = t[i-1] + gap + eps
            t[i] = max(t[i], lo)
    return t.tolist()

def repair_one_uav(name, v, th, t123, z123):
    """单 UAV 修复：边界、角度、时间排序+≥1s、tau 上界（含物理 Ez>0）"""
    v = float(np.clip(v, 70.0, 140.0))
    th = wrap_deg(th)
    # 时间修复
    t_sorted = enforce_min_gap_sorted(t123, gap=MIN_GAP, eps=EPS, tmax=T_HIT_MAX)
    # tau 上界
    tau_max = TAU_MAXS[name]
    z = np.clip(np.array(z123, dtype=float), 0.0, tau_max)
    return v, th, t_sorted, z.tolist()

def repair_vector_q5(x):
    """
    x 结构（共 40 维）：
    [ v1,th1,t11,t12,t13,z11,z12,z13,  v2,th2,t21,t22,t23,z21,z22,z23,  ... × 5 UAV ]
    """
    x = np.array(x, dtype=float).tolist()
    out = []
    idx = 0
    for uav in UAV_NAMES:
        v = x[idx+0]; th = x[idx+1]
        t123 = [x[idx+2], x[idx+3], x[idx+4]]
        z123 = [x[idx+5], x[idx+6], x[idx+7]]
        v, th, t_sorted, z = repair_one_uav(uav, v, th, t123, z123)
        out += [v, th] + t_sorted + z
        idx += 8
    return np.array(out, dtype=float)

def build_uavs_from_x(xr):
    """把 40 维向量构造成 5 架 UAV 的参数列表"""
    uavs = []
    idx = 0
    for uav in UAV_NAMES:
        v = float(xr[idx+0]); th = float(xr[idx+1])
        t123 = [float(xr[idx+2]), float(xr[idx+3]), float(xr[idx+4])]
        z123 = [float(xr[idx+5]), float(xr[idx+6]), float(xr[idx+7])]
        uavs.append(dict(
            name=uav, F0=FY0[uav], v=v, theta_deg=th, t_drop=t123, tau=z123
        ))
        idx += 8
    return uavs

# =========================
# 3) 目标函数（最小化）
# =========================
def obj_neg_cover_q5(x, dt=EVAL_DT_COARSE, print_debug=False):
    """
    目标 = - (sum_per_missile_cover + LAMBDA * intersection_cover)
    """
    xr = repair_vector_q5(x)
    uavs = build_uavs_from_x(xr)
    per_union, per_cov, cov_sum, inter_union, inter_cov, _ = evaluate_plan(uavs, dt=dt, print_details=False)
    score = cov_sum + LAMBDA_INTERSECT * inter_cov
    if print_debug:
        print("per_cov:", per_cov, "sum:", cov_sum, "inter:", inter_cov, "-> score", score)
    return -float(score), xr

# =========================
# 4) 粒子群算法（PSO）
# =========================
class PSO:
    def __init__(self, dim, bounds, pop_size=220, iters=1000,
                 w_start=0.90, w_end=1, c1=1.7, c2=1.7,
                 v_max_scale=0.25):
        """
        bounds: list[(lo,hi)] per dim
        v_max_scale: 速度上限 = v_max_scale * (hi-lo)
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
            # 角度列 wrap（每 8 维的第 1 列为角度）
            for b in range(5):
                k = b*8 + 1
                xi[k] = wrap_deg(xi[k])
            X.append(xi)
        self.X = np.vstack(X)
        self.V = self.rng.normal(0, 0.1, size=(pop_size, dim)) * self.vmax
        self.P = self.X.copy()
        self.Pf = np.full(pop_size, np.inf, dtype=float)
        self.G = None; self.Gf = np.inf
        self.Gx_repaired = None

    def clamp(self, X):
        return np.clip(X, self.bounds[:,0], self.bounds[:,1])

    def run(self, objective, dt_eval=EVAL_DT_COARSE, verbose=True):
        for it in range(1, self.iters+1):
            # 评估与最优更新
            for i in range(self.pop_size):
                f, xr = objective(self.X[i], dt=dt_eval)
                if f < self.Pf[i]:
                    self.Pf[i] = f; self.P[i] = self.X[i].copy()
                if f < self.Gf:
                    self.Gf = f; self.G = self.X[i].copy(); self.Gx_repaired = xr.copy()
            # 更新
            w = self.w_start + (self.w_end - self.w_start) * (it / self.iters)
            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))
            cognitive = self.c1 * r1 * (self.P - self.X)
            social    = self.c2 * r2 * (self.G - self.X)
            self.V = w*self.V + cognitive + social
            self.V = np.clip(self.V, -self.vmax, self.vmax)
            self.X = self.clamp(self.X + self.V)

            if verbose and (it % max(1, self.iters//10) == 0):
                print(f"[PSO] iter={it:4d}  best_obj={self.Gf:.6f}  est_score={-self.Gf:.6f}")

        return self.Gx_repaired, self.Gf

# =========================
# 5) 导出
# =========================
def export_result(filename, uavs, per_missile_union, per_missile_cover,
                  inter_union, inter_cover, per_bomb_rows, score_sum):
    try:
        import pandas as pd
        with pd.ExcelWriter(filename, engine="xlsxwriter") as w:
            # Summary
            summ = [{
                "score=sum(cover_i)+λ*inter": score_sum,
                "lambda": LAMBDA_INTERSECT,
                "M1_cover": per_missile_cover["M1"],
                "M2_cover": per_missile_cover["M2"],
                "M3_cover": per_missile_cover["M3"],
                "intersection_cover": inter_cover
            }]
            pd.DataFrame(summ).to_excel(w, index=False, sheet_name="summary")

            # UAV 参数
            rows = []
            for u in uavs:
                rows.append({
                    "uav": u["name"], "v": u["v"], "theta_deg": u["theta_deg"],
                    "t1": u["t_drop"][0], "t2": u["t_drop"][1], "t3": u["t_drop"][2],
                    "tau1": u["tau"][0], "tau2": u["tau"][1], "tau3": u["tau"][2],
                })
            pd.DataFrame(rows).to_excel(w, index=False, sheet_name="uav_params")

            # 每个导弹的并集区间
            for m in ["M1","M2","M3"]:
                ivs = per_missile_union[m]
                df = pd.DataFrame([{"t_start": a, "t_end": b, "dur": b-a} for (a,b) in ivs])
                if df.empty: df = pd.DataFrame(columns=["t_start","t_end","dur"])
                df.to_excel(w, index=False, sheet_name=f"{m}_union")

            # 三导弹交集
            dfI = pd.DataFrame([{"t_start": a, "t_end": b, "dur": b-a} for (a,b) in inter_union])
            if dfI.empty: dfI = pd.DataFrame(columns=["t_start","t_end","dur"])
            dfI.to_excel(w, index=False, sheet_name="intersection")

            # per_bomb（每弹对每导弹一行）
            pd.DataFrame(per_bomb_rows).to_excel(w, index=False, sheet_name="per_bomb")

        print(f"[导出] 已写入 {filename}")
    except Exception as e:
        print(f"[导出] 写入 {filename} 失败：{e}")

# =========================
# 6) 主入口
# =========================
def main():
    t0 = time.time()

    # 变量向量维度与边界（40 维）
    dim = 8 * 5
    bounds = []
    for uav in UAV_NAMES:
        tau_max = TAU_MAXS[uav]
        # [v, theta, t1, t2, t3, z1, z2, z3]
        bounds += [
            (70.0, 140.0),      # v
            (-180.0, 180.0),    # theta
            (0.0, T_HIT_MAX),   # t1
            (0.0, T_HIT_MAX),   # t2
            (0.0, T_HIT_MAX),   # t3
            (0.0, tau_max),     # z1
            (0.0, tau_max),     # z2
            (0.0, tau_max),     # z3
        ]

    # PSO 超参（可接入你之前的强化调参器）
    pso = PSO(dim, bounds,
              pop_size=220, iters=2600,
              w_start=0.90, w_end=1,
              c1=1.7, c2=1.7, v_max_scale=0.25)

    # 运行（粗评）
    best_xr, best_obj = pso.run(obj_neg_cover_q5, dt_eval=EVAL_DT_COARSE, verbose=True)
    est_score = -best_obj

    # 复评与导出
    uavs = build_uavs_from_x(best_xr)
    per_union, per_cov, cov_sum, inter_union, inter_cov, per_bomb_rows = evaluate_plan(
        uavs, dt=EVAL_DT_FINE, print_details=True
    )
    score_sum = cov_sum + LAMBDA_INTERSECT * inter_cov

    t1 = time.time()

    # 打印
    print("\n=== 问题5 最终方案（5 UAV × 3 枚, PSO） ===")
    print(f"[估计得分(粗)] {est_score:.6f} = sum(cover)+λ*inter")
    print(f"[细评得分   ] {score_sum:.6f}  (λ={LAMBDA_INTERSECT})")
    for k in ["M1","M2","M3"]:
        print(f"  · {k} 并集遮蔽 = {per_cov[k]:.6f} s")
    print(f"  · 同时遮蔽(交集) = {inter_cov:.6f} s")
    for u in uavs:
        print(f"{u['name']}: v={u['v']:.3f}, θ={u['theta_deg']:.3f}, "
              f"t={list(map(lambda x:f'{x:.3f}', u['t_drop']))}, "
              f"τ={list(map(lambda x:f'{x:.3f}', u['tau']))}")
    print(f"[耗时] {t1 - t0:.3f} s")

    # 导出
    export_result("result3.xlsx", uavs, per_union, per_cov, inter_union, inter_cov, per_bomb_rows, score_sum)

if __name__ == "__main__":
    main()
