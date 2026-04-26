# -*- coding: utf-8 -*-
"""
Problem 4 —— 两架无人机协同（FY1 & FY2），总投放次数受限，最大化对 M1 的遮蔽并集时长
参照：你的第3问算法（网格 -> 候选 -> 贪心 -> 局部微调）的结构扩展到两机协同
特性：
- 体积目标：圆柱体（半径 7 m，高 10 m）
- 遮蔽口径：MODE='ANY' | 'ALL'
- 约束：同一无人机相邻投放间隔 >= MIN_GAP（默认 1s）
- 输出：result4.xlsx（自动选择 xlsxwriter/openpyxl；都无时降级 CSV）

依赖：numpy, pandas
"""

import math
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict

# =========================
# 数值/几何/区间工具
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
    扫描 f(t)=D(t)-R_cloud，返回 D<=R 的所有区间
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

def merge_intervals(intervals: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9: out[-1] = (la, max(lb, b))
        else: out.append((a,b))
    return out

def total_length(intervals: List[Tuple[float,float]]) -> float:
    return sum(b-a for a,b in intervals)

# =========================
# 体积目标（圆柱）采样
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

# M1：直指假目标原点
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

# FY1, FY2 初始（若题面不同请改）
FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([17800.0,  800.0, 1800.0])

v_min, v_max = 70.0, 140.0

def drone_pos(F0, t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(F0, v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(F0, t_drop, v_u, theta)
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

def evaluate_single(F0, v_u, theta, t_drop, tau, mode='ANY', scan_dt=0.06):
    if not (v_min <= v_u <= v_max): return [], None
    R, E = explosion_point(F0, v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return [], None  # 起爆点高度>0
    t_e = t_drop + tau
    C = cloud_center_builder(E, t_e)
    f = build_volume_cover_fn(CYL_PTS, missile_pos, C, mode=mode)
    intervals = find_cover_intervals(f, t_e, t_e + effective_span, dt=scan_dt)
    return intervals, dict(R=R, E=E, t_e=t_e)

# =========================
# 候选生成（每机独立），并按单枚时长 TopK 裁剪
# =========================
def make_candidates_for_uav(uav_name, F0, v_u, theta,
                            drop_grid, tau_grid, mode='ANY',
                            scan_dt=0.06, topk=600):
    cand = []
    for t_d in drop_grid:
        for tau in tau_grid:
            ivs, info = evaluate_single(F0, v_u, theta, t_d, tau, mode=mode, scan_dt=scan_dt)
            if not ivs:
                continue
            cand.append({
                "uav": uav_name,
                "F0": F0,
                "v": v_u,
                "theta": theta,
                "t_d": float(t_d),
                "tau": float(tau),
                "ivs": ivs,
                "len": total_length(ivs),
                "info": info
            })
    if not cand: return []
    cand.sort(key=lambda c: c["len"], reverse=True)
    return cand[:topk]

# =========================
# 全局贪心（跨两机），并集最大 + 同机最小间隔约束
# =========================
def greedy_select_global(all_cands: List[Dict], N_total: int, min_gap: float) -> Tuple[float, List[Dict], List[Tuple[float,float]]]:
    chosen: List[Dict] = []
    covered: List[Tuple[float,float]] = []
    last_times: Dict[str, List[float]] = {}  # 每 UAV 已选投放时刻

    def feasible(c):
        times = last_times.get(c["uav"], [])
        return all(abs(c["t_d"] - t) >= min_gap for t in times)

    def marginal_gain(ivs):
        merged = merge_intervals(covered + ivs)
        return total_length(merged) - total_length(covered)

    # 先按“单枚时长”排序，提升早期选取质量
    all_cands = sorted(all_cands, key=lambda x: x["len"], reverse=True)

    for _ in range(N_total):
        best, best_gain = None, 0.0
        for c in all_cands:
            if not feasible(c):
                continue
            g = marginal_gain(c["ivs"])
            if g > best_gain + 1e-9:
                best, best_gain = c, g
        if best is None:
            # 无正增益候选时，选一个可行的“最长单枚”填充
            for c in all_cands:
                if feasible(c):
                    best, best_gain = c, 0.0
                    break
        if best is None:
            break
        chosen.append(best)
        covered = merge_intervals(covered + best["ivs"])
        last_times.setdefault(best["uav"], []).append(best["t_d"])
        # 移除与该 UAV 时间冲突的候选（降复杂度）
        all_cands = [c for c in all_cands if c is best or c["uav"] != best["uav"] or abs(c["t_d"] - best["t_d"]) >= min_gap]

    return total_length(covered), chosen, covered

# =========================
# 局部连续微调（固定两机的 θ/v，仅调每枚 t_d & tau）
# =========================
def local_refine(chosen: List[Dict], union_iv: List[Tuple[float,float]], iters=500, seed=123, min_gap=1.0, mode='ANY'):
    rng = np.random.default_rng(seed)

    def eval_set(items: List[Dict]):
        # 按 UAV 校验时间间隔
        by_uav: Dict[str, List[float]] = {}
        for c in items:
            by_uav.setdefault(c["uav"], []).append(c["t_d"])
        for u, ts in by_uav.items():
            ts = sorted(ts)
            for i in range(1, len(ts)):
                if ts[i] - ts[i-1] < min_gap:
                    return -1.0, []  # 不可行

        # 重新评估每枚（更细 dt），并合并并集
        ivs_all = []
        new = []
        for c in items:
            ivs, info = evaluate_single(c["F0"], c["v"], c["theta"], c["t_d"], c["tau"], mode=mode, scan_dt=0.04)
            if not ivs:
                return -1.0, []
            cc = dict(c); cc["ivs"] = ivs; cc["info"] = info
            new.append(cc); ivs_all += ivs
        merged = merge_intervals(ivs_all)
        return total_length(merged), merged

    best_set = [dict(c) for c in chosen]
    best_val  = total_length(union_iv)
    best_union = union_iv

    for _ in range(iters):
        cand = [dict(c) for c in best_set]
        # 小扰动
        for c in cand:
            c["t_d"] = max(0.0, c["t_d"] + rng.normal(0, 0.8))
            c["tau"] = max(0.25, c["tau"] + rng.normal(0, 0.5))
        val, un = eval_set(cand)
        if val > best_val + 1e-9:
            best_val, best_union, best_set = val, un, cand

    return best_val, best_set, best_union

# =========================
# 自适应写文件（xlsxwriter/openpyxl/CSV 降级）
# =========================
def safe_write_result(out_path, plan_df, union_intervals, summary_dict):
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".xlsx":
        out_path = root + ".xlsx"

    def try_with_engine(engine_name):
        try:
            with pd.ExcelWriter(out_path, engine=engine_name) as w:
                plan_df.to_excel(w, index=False, sheet_name="plan")
                pd.DataFrame([{"union_intervals": str(union_intervals)}]).to_excel(w, index=False, sheet_name="union_info")
                pd.DataFrame([summary_dict]).to_excel(w, index=False, sheet_name="summary")
            print(f"[OK] 结果已用 {engine_name} 写入: {out_path}")
            return True
        except ModuleNotFoundError as e:
            if engine_name in str(e):
                print(f"[WARN] 未安装 {engine_name}，尝试其它引擎…")
                return False
            raise

    if try_with_engine("xlsxwriter"): return out_path
    if try_with_engine("openpyxl"):   return out_path

    # 降级 CSV
    plan_df.to_csv(root + "_plan.csv", index=False)
    pd.DataFrame([{"union_intervals": str(union_intervals)}]).to_csv(root + "_union_info.csv", index=False)
    pd.DataFrame([summary_dict]).to_csv(root + "_summary.csv", index=False)
    print(f"[FALLBACK] 无 xlsxwriter/openpyxl，已降级 CSV 到 {root}_*.csv")
    return root + "_plan.csv"

# =========================
# 主流程：枚举两机 θ/v → 生成候选 → 全局贪心 → 局部微调 → 导出
# =========================
def solve_problem4(mode="ANY"):
    # —— 参数区（你可按算力调节）——
    HEADING_SET = np.linspace(0.0, 2.0*math.pi, 24, endpoint=False)  # 每机 24 个航向
    SPEED_SET   = [90.0, 100.0, 110.0, 120.0, 130.0]                 # 每机 5 档速度
    DROP_GRID   = np.arange(0.0, 60.0 + 1e-9, 2.0)                   # 投放时刻
    TAU_GRID    = np.arange(1.0, 9.0  + 1e-9, 1.0)                   # 引信延时
    MIN_GAP     = 1.0                                                # 同机间隔
    N_TOTAL     = 6                                                  # 总投放上限
    TOPK_EACH   = 600                                                # 每机候选截断

    best_overall = None

    for th1 in HEADING_SET:
        for v1 in SPEED_SET:
            # 先预生成 FY1 候选（可重复用）
            cands1 = make_candidates_for_uav("FY1", FY1_F0, v1, th1, DROP_GRID, TAU_GRID, mode=mode, scan_dt=0.06, topk=TOPK_EACH)
            if not cands1:
                continue
            for th2 in HEADING_SET:
                for v2 in SPEED_SET:
                    cands2 = make_candidates_for_uav("FY2", FY2_F0, v2, th2, DROP_GRID, TAU_GRID, mode=mode, scan_dt=0.06, topk=TOPK_EACH)
                    if not cands2:
                        continue
                    # 全局贪心选择 N_TOTAL 枚
                    tot, chosen, union_iv = greedy_select_global(cands1 + cands2, N_TOTAL, MIN_GAP)
                    if best_overall is None or tot > best_overall[0]:
                        best_overall = (tot, th1, v1, th2, v2, chosen, union_iv)

    if best_overall is None:
        print("未找到可行方案；可加密网格或放宽 TOPK/步长。")
        return None

    base_total, th1, v1, th2, v2, chosen, union_iv = best_overall
    print(f"[粗解] 并集时长≈{base_total:.3f}s | FY1(θ={((th1*180/math.pi)%360):.1f}°, v={v1})  FY2(θ={((th2*180/math.pi)%360):.1f}°, v={v2}) | 选 {len(chosen)} 枚")

    # 局部微调（仅调 t_d 和 tau）
    ref_total, refined, refined_union = local_refine(chosen, union_iv, iters=600, seed=123, min_gap=MIN_GAP, mode=mode)
    if ref_total > base_total:
        chosen, union_iv, total_best = refined, refined_union, ref_total
    else:
        total_best = base_total

    # 导出
    rows = []
    for c in sorted(chosen, key=lambda x: (x["uav"], x["t_d"])):
        info = c["info"]; R, E, t_e = info["R"], info["E"], info["t_e"]
        segs = sorted(c["ivs"], key=lambda s: (s[1]-s[0]), reverse=True)
        tin, tout, dur = (segs[0][0], segs[0][1], segs[0][1]-segs[0][0]) if segs else (float('nan'),)*3
        rows.append({
            "uav": c["uav"],
            "heading_deg": (c["theta"]*180.0/math.pi)%360.0,
            "speed_mps": c["v"],
            "drop_time_s": c["t_d"],
            "fuse_delay_s": c["tau"],
            "R_x_m": R[0], "R_y_m": R[1], "R_z_m": R[2],
            "E_x_m": E[0], "E_y_m": E[1], "E_z_m": E[2],
            "t_explode_s": t_e,
            "cover_tin_s": tin, "cover_tout_s": tout, "cover_duration_s": dur
        })
    df = pd.DataFrame(rows)

    summary = {
        "uav": "SUMMARY",
        "FY1_heading_deg": (th1*180.0/math.pi)%360.0, "FY1_speed_mps": v1,
        "FY2_heading_deg": (th2*180.0/math.pi)%360.0, "FY2_speed_mps": v2,
        "mode": mode, "cover_duration_s": total_best,
        "N_total": len(chosen)
    }

    out_path = "result4.xlsx"
    safe_write_result(out_path, df, union_iv, summary)
    print(f"[OK] 导出完成 -> {out_path}")
    print(df.to_string(index=False))
    return out_path, df, union_iv, total_best

# =========================
# 直接运行
# =========================
if __name__ == "__main__":
    MODE = "ALL"   # 可改 "ALL"
    solve_problem4(mode=MODE)
