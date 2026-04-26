# -*- coding: utf-8 -*-
"""
Problem 3 (FY1 drops 3 charges): maximize union of cover time against M1
with volume target (cylindrical) occlusion test. ANY/ALL modes supported.

Output: result1.xlsx (plan / union_info / summary)
"""

import math
import numpy as np
import pandas as pd

# =========================
# 工具函数（数值/几何/区间）
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

def find_cover_intervals(f, t0, t1, dt=0.06):
    """
    在 [t0, t1] 扫描 f(t)=D(t)-R_cloud，返回所有满足 D<=R 的区间列表
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
            if r is not None:
                roots.append(r)
    roots = sorted(set(roots))

    def inside(x): return f(x) <= 0.0
    intervals = []
    cur_in, cursor = inside(t0), t0
    for r in roots:
        if cur_in:
            intervals.append((cursor, r)); cur_in = False
        else:
            cursor, cur_in = r, True
    if cur_in:
        intervals.append((cursor, t1))

    return [(a,b) for a,b in intervals if b > a + 1e-8]

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9: out[-1] = (la, max(lb, b))
        else: out.append((a,b))
    return out

def total_length(intervals): return sum(b-a for a,b in intervals)

# =========================
# 目标（体积）圆柱体采样
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0
TARGET_H = 10.0

def sample_cylinder_points(base, radius, height,
                           n_theta_side=64, n_z_side=8,
                           n_theta_disk=48, n_r_disk=3):
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
    # 上/下底
    for z in [0.0, height]:
        thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(0.0, radius, n_r_disk+1)[1:]  # 去 r=0
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

# =========================
# 物理参数（导弹/无人机/云团）
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# M1 导弹（直指假目标原点）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

# FY1 初始
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
    else:
        def f(t):
            M = missile_pos_func(t); C = cloud_center_func(t)
            Ms = np.repeat(M[None,:], len(Xi), axis=0)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            return float(np.max(d) - R_cloud)
        return f

# =========================
# 单枚弹评估（体积目标）
# =========================
def evaluate_cover_intervals(v_u, theta, t_drop, tau, mode='ANY', scan_dt=0.06):
    if not (v_min <= v_u <= v_max): return [], None
    R, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return [], None  # 起爆高度>0
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    intervals = find_cover_intervals(f, t_e, t_e + effective_span, dt=scan_dt)
    return intervals, dict(R=R, E=E, t_e=t_e)

# =========================
# 贪心选 3 枚（给定 θ, v），并集最大
# =========================
def greedy_select_three(theta, v_u, drop_grid, tau_grid, mode='ANY', min_gap=1.0):
    candidates = []
    for t_d in drop_grid:
        for tau in tau_grid:
            ivs, info = evaluate_cover_intervals(v_u, theta, t_d, tau, mode=mode, scan_dt=0.06)
            if not ivs:  # 无遮蔽则不纳入
                continue
            candidates.append({
                "t_d": float(t_d),
                "tau": float(tau),
                "ivs": ivs,
                "len": total_length(ivs),
                "info": info
            })
    if not candidates:
        return 0.0, [], []

    # 先按“单枚时长”排序，裁掉尾部以控规模
    candidates.sort(key=lambda c: c["len"], reverse=True)
    candidates = candidates[:800]

    chosen, covered = [], []
    def marginal_gain(ivs):
        merged = merge_intervals(covered + ivs)
        return total_length(merged) - total_length(covered)

    # 阶段1：基于边际增益的贪心（含间隔约束）
    while len(chosen) < 3:
        best, best_gain = None, 0.0
        for c in candidates:
            if any(abs(c["t_d"] - cc["t_d"]) < min_gap for cc in chosen):  # 投放间隔≥1 s
                continue
            g = marginal_gain(c["ivs"])
            if g > best_gain + 1e-9:
                best, best_gain = c, g
        if best is None or best_gain <= 1e-9:
            break
        chosen.append(best)
        covered = merge_intervals(covered + best["ivs"])

    # 阶段2：若不足 3 枚，按“单枚最长”补齐（满足间隔约束），即便边际增益为 0
    if len(chosen) < 3:
        for c in candidates:
            if any(abs(c["t_d"] - cc["t_d"]) < min_gap for cc in chosen):
                continue
            chosen.append(c)
            covered = merge_intervals(covered + c["ivs"])
            if len(chosen) >= 3:
                break

    return total_length(covered), chosen[:3], covered

# =========================
# 局部连续微调（在已选三枚的邻域）
# =========================
def local_refine_triple(theta, v_u, triple, mode='ANY', iters=400, seed=0, min_gap=1.0):
    """
    triple: list of dicts [{"t_d":..,"tau":..}, *3]
    只在 (t_d, tau) 连续域做微调（θ、v 固定），保持起爆高度>0和投放间隔约束
    """
    rng = np.random.default_rng(seed)

    def eval_union(tr):
        ivs_all = []
        for c in tr:
            ivs, info = evaluate_cover_intervals(v_u, theta, c["t_d"], c["tau"], mode=mode, scan_dt=0.04)
            if not ivs:
                return -1.0, []  # 不可行/无覆盖
            c["ivs"], c["info"] = ivs, info
            ivs_all += ivs
        merged = merge_intervals(ivs_all)
        return total_length(merged), merged

    # 初始
    cur = [{"t_d":c["t_d"], "tau":c["tau"]} for c in triple]
    # 确保间隔
    cur = sorted(cur, key=lambda x: x["t_d"])
    ok_intervals = all(cur[i+1]["t_d"] - cur[i]["t_d"] >= min_gap for i in range(2))
    if not ok_intervals:  # 若输入三枚间隔未满足，强行略微拉开
        for i in range(1,3):
            cur[i]["t_d"] = max(cur[i]["t_d"], cur[i-1]["t_d"] + min_gap)

    best_val, best_union = eval_union(cur)
    best_triple = [dict(x) for x in cur]

    for k in range(iters):
        scale = 1.0 + 0.5 * rng.random()
        cand = [dict(x) for x in best_triple]
        # 对三枚分别扰动（较小步长）
        for i in range(3):
            cand[i]["t_d"] = max(0.0, cand[i]["t_d"] + rng.normal(0, 0.8/scale))
            cand[i]["tau"] = max(0.25, cand[i]["tau"] + rng.normal(0, 0.5/scale))
        cand = sorted(cand, key=lambda x: x["t_d"])
        # 再次强制间隔
        for i in range(1,3):
            if cand[i]["t_d"] - cand[i-1]["t_d"] < min_gap:
                cand[i]["t_d"] = cand[i-1]["t_d"] + min_gap

        val, un = eval_union(cand)
        if val > best_val:
            best_val, best_union = val, un
            best_triple = cand

    # 回填详细信息
    final = []
    for c in best_triple:
        ivs, info = evaluate_cover_intervals(v_u, theta, c["t_d"], c["tau"], mode=mode, scan_dt=0.03)
        final.append({"t_d":c["t_d"], "tau":c["tau"], "ivs":ivs, "info":info})
    return best_val, final, best_union

# =========================
# 主流程：搜索 θ,v ，贪心选三枚 + 局部微调
# =========================
def solve_problem3(mode="ANY"):
    # 搜索网格（可加密/放宽）
    thetas = np.linspace(0.0, 2.0*math.pi, 100, endpoint=False)  # 航向角
    v_candidates = [ 115.0,120.0,125.0]           # 速度档
    drop_grid = np.arange(0.0, 60.0 + 1e-9, 2.0)                # 投放时刻
    tau_grid  = np.arange(1.0, 9.0 + 1e-9, 1.0)                 # 延时
    min_gap   = 1.0

    best = None
    for theta in thetas:
        for v_u in v_candidates:
            total, chosen, union_intervals = greedy_select_three(theta, v_u, drop_grid, tau_grid,
                                                                mode=mode, min_gap=min_gap)
            if best is None or total > best[0]:
                best = (total, theta, v_u, chosen, union_intervals)

    if best is None or best[0] <= 0:
        print("未找到可行三弹方案（可加密网格或放宽搜索范围）。")
        return None

    base_total, theta, v_u, chosen, union_intervals = best

    # 局部连续微调（在 chosen 的邻域上精化 t_drop 与 τ）
    seed = 123
    triple0 = [{"t_d":c["t_d"], "tau":c["tau"]} for c in chosen]
    ref_total, refined, refined_union = local_refine_triple(theta, v_u, triple0,
                                                            mode=mode, iters=500,
                                                            seed=seed, min_gap=min_gap)

    total_best = max(base_total, ref_total)
    final = refined if ref_total >= base_total else chosen
    union_final = refined_union if ref_total >= base_total else union_intervals

    # 整理导出
    final_sorted = sorted(final, key=lambda c: c["t_d"])
    rows = []
    for c in final_sorted:
        info = c["info"]
        R = info["R"]; E = info["E"]; t_e = info["t_e"]
        # 展示每枚的最长遮蔽段
        segs = sorted(c["ivs"], key=lambda s: (s[1]-s[0]), reverse=True)
        if segs:
            tin, tout = segs[0]; dur = tout - tin
        else:
            tin = tout = dur = float('nan')

        rows.append({
            "uav": "FY1",
            "heading_deg": (theta * 180.0 / math.pi) % 360.0,
            "speed_mps": v_u,
            "drop_time_s": c["t_d"],
            "fuse_delay_s": c["tau"],
            "R_x_m": R[0], "R_y_m": R[1], "R_z_m": R[2],
            "E_x_m": E[0], "E_y_m": E[1], "E_z_m": E[2],
            "t_explode_s": t_e,
            "cover_tin_s": tin, "cover_tout_s": tout, "cover_duration_s": dur
        })

    df = pd.DataFrame(rows)
    out_path = "result1.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="plan")
        pd.DataFrame([{"union_intervals": str(union_final)}]).to_excel(w, index=False, sheet_name="union_info")
        pd.DataFrame([{
            "uav":"SUMMARY",
            "heading_deg": df["heading_deg"].iloc[0] if len(df)>0 else np.nan,
            "speed_mps": df["speed_mps"].iloc[0] if len(df)>0 else np.nan,
            "cover_duration_s": total_best,
            "mode": mode
        }]).to_excel(w, index=False, sheet_name="summary")

    print(f"[OK] 方案导出 -> {out_path}")
    print(f"航向θ≈{((theta*180.0/math.pi)%360.0):.2f}°  速度v≈{v_u:.2f} m/s  "
          f"三弹并集时长≈{total_best:.3f} s  （模式：{mode}）")
    print(df.to_string(index=False))
    return out_path, df, union_final, total_best

# =========================
# 直接运行
# =========================
if __name__ == "__main__":
    MODE = "ALL"   # 可改 "ALL"
    solve_problem3(mode=MODE)
