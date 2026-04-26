import math
import numpy as np
import pandas as pd

# ============ 几何与数值工具 ============
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def dist_point_to_segment(P, A, B):
    BA = B - A
    l2 = float(np.dot(BA, BA))
    if l2 == 0.0:
        return float(np.linalg.norm(P - A)), 0.0
    s = float(np.dot(P - A, BA) / l2)
    s = max(0.0, min(1.0, s))
    Q = A + s * BA
    return float(np.linalg.norm(P - Q)), s

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
        if fa * fm <= 0: hi, fb = mid, fm
        else: lo, fa = mid, fm
    return 0.5 * (lo + hi)

def find_cover_intervals(f, t0, t1, dt=0.08):
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
            r = bisect_root(f, a, b);  roots.append(r) if r is not None else None
    roots = sorted(roots)

    def inside(t): return f(t) <= 0.0
    intervals, cur_in, cursor = [], inside(t0), t0
    for r in roots:
        if cur_in: intervals.append((cursor, r)); cur_in = False
        else: cursor, cur_in = r, True
    if cur_in: intervals.append((cursor, t1))
    return [(a,b) for a,b in intervals if b > a + 1e-6]

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

# ============ 题面常量（A题） ============
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0

# 真目标/导弹初始/无人机初始
T  = np.array([0.0, 200.0, 0.0])                 # 真目标
M0 = np.array([20000.0, 0.0, 2000.0])            # M1
F0 = np.array([17800.0, 0.0, 1800.0])            # FY1
u_m = unit(-M0)
v_min, v_max = 70.0, 140.0

# 轨迹与遮蔽函数
def missile_pos(t): return M0 + v_m * t * u_m

def drone_pos(t, v_u, theta):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    return F0 + v_u * t * h

def explosion_point(v_u, theta, t_drop, tau):
    h = np.array([math.cos(theta), math.sin(theta), 0.0])
    R = drone_pos(t_drop, v_u, theta)
    E = R + v_u * tau * h + 0.5 * np.array([0.0,0.0,-g]) * (tau**2)
    return R, E

def make_D_minus_R(v_u, theta, t_drop, tau):
    R, E = explosion_point(v_u, theta, t_drop, tau)
    t_e = t_drop + tau
    def cloud_center(t): return E + np.array([0.0,0.0,-sink_v]) * (t - t_e)
    def f_scalar(t):
        A, B, P = missile_pos(t), T, cloud_center(t)
        D, _ = dist_point_to_segment(P, A, B)
        return D - R_cloud
    return f_scalar, t_e, R, E

def evaluate_cover_intervals(v_u, theta, t_drop, tau, scan_dt=0.08):
    if not (v_min <= v_u <= v_max): return [], None
    _, E = explosion_point(v_u, theta, t_drop, tau)
    if E[2] <= 0.0: return [], None
    f_scalar, t_e, R, E = make_D_minus_R(v_u, theta, t_drop, tau)
    intervals = find_cover_intervals(f_scalar, t_e, t_e + effective_span, dt=scan_dt)
    return intervals, dict(R=R, E=E, t_e=t_e)

# ============ 问题3 求解（贪心 + 补齐到 3 枚） ============
def greedy_three(theta, v_u, drop_grid, tau_grid, min_gap=1.0):
    cands = []
    for t_d in drop_grid:
        for tau in tau_grid:
            ivs, info = evaluate_cover_intervals(v_u, theta, t_d, tau, scan_dt=0.08)
            if not ivs:  # 没遮蔽的候选跳过（补齐阶段另行处理）
                continue
            cands.append({"t_d":float(t_d), "tau":float(tau), "ivs":ivs,
                          "len":total_length(ivs), "info":info})
    if not cands: return 0.0, [], []

    cands.sort(key=lambda c: c["len"], reverse=True)
    cands = cands[:300]  # 限制规模

    chosen, covered = [], []
    def gain(ivs): return total_length(merge_intervals(covered + ivs)) - total_length(covered)

    # 阶段1：按边际增益贪心
    while len(chosen) < 3:
        best, best_gain = None, 0.0
        for c in cands:
            if any(abs(c["t_d"] - cc["t_d"]) < min_gap for cc in chosen): continue
            g = gain(c["ivs"])
            if g > best_gain + 1e-9: best, best_gain = c, g
        if best is None or best_gain <= 1e-9: break
        chosen.append(best); covered = merge_intervals(covered + best["ivs"])

    # 阶段2：若不足 3 枚，用“单枚时长最大”的候选补齐（即便边际增益为0）
    if len(chosen) < 3:
        for c in cands:
            if any(abs(c["t_d"] - cc["t_d"]) < min_gap for cc in chosen): continue
            chosen.append(c); covered = merge_intervals(covered + c["ivs"])
            if len(chosen) >= 3: break

    return total_length(covered), chosen[:3], covered

def solve_problem3():
    # 轻量网格（可加密）：6 个航向 × 3 档速度 ×（投放 0..40 每 4s）×（延时 1..7 每 1s）
    thetas = np.linspace(0.0, 2.0*math.pi, 6, endpoint=False)
    vset   = [80.0, 110.0, 140.0]
    drops  = np.arange(0.0, 40.0+1e-9, 4.0)
    taus   = np.arange(1.0, 7.0+1e-9, 1.0)

    best = None
    for th in thetas:
        for v in vset:
            total, chosen, u = greedy_three(th, v, drops, taus, min_gap=1.0)
            if best is None or total > best[0]:
                best = (total, th, v, chosen, u)

    # 若不足 3 枚带增益的候选，尝试在最佳(th, v)附近再搜一个能覆盖的第三枚以“补齐 3 枚”
    total, th, v, chosen, union = best
    if len(chosen) < 3:
        exist_tds = [c["t_d"] for c in chosen]
        best_extra, best_len = None, 0.0
        for t_d in np.arange(6.0, 40.0+1e-9, 1.0):
            if any(abs(t_d - td) < 1.0 for td in exist_tds): continue
            for tau in np.arange(1.0, 9.0+1e-9, 0.5):
                ivs, info = evaluate_cover_intervals(v, th, t_d, tau, scan_dt=0.06)
                L = total_length(ivs)
                if info is not None and L > best_len:
                    best_len, best_extra = L, {"t_d":float(t_d),"tau":float(tau),"ivs":ivs,"len":L,"info":info}
        if best_extra:
            chosen.append(best_extra)
            union = merge_intervals(union + best_extra["ivs"])
            total = total_length(union)

    # 整理与导出
    chosen = sorted(chosen, key=lambda c: c["t_d"])
    rows = []
    for c in chosen:
        info = c["info"]; R, E, t_e = info["R"], info["E"], info["t_e"]
        segs = sorted(c["ivs"], key=lambda s: (s[1]-s[0]), reverse=True)
        tin, tout, dur = (segs[0][0], segs[0][1], segs[0][1]-segs[0][0]) if segs else (np.nan,np.nan,np.nan)
        rows.append({
            "uav":"FY1","heading_deg": (th*180.0/math.pi)%360.0,"speed_mps": v,
            "drop_time_s": c["t_d"], "fuse_delay_s": c["tau"],
            "R_x_m": R[0], "R_y_m": R[1], "R_z_m": R[2],
            "E_x_m": E[0], "E_y_m": E[1], "E_z_m": E[2],
            "t_explode_s": t_e,
            "cover_tin_s": tin, "cover_tout_s": tout, "cover_duration_s": dur
        })
    df = pd.DataFrame(rows)
    out_path = "result1.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="plan")
        pd.DataFrame([{"union_intervals": str(union)}]).to_excel(w, index=False, sheet_name="union_info")
        pd.DataFrame([{"uav":"SUMMARY","heading_deg":df["heading_deg"].iloc[0],"speed_mps":df["speed_mps"].iloc[0],
                       "cover_duration_s": total}]).to_excel(w, index=False, sheet_name="summary")
    print(f"[OK] FY1 三弹方案已导出到 {out_path}；遮蔽并集总时长 ≈ {total:.3f} s")
    print(df.to_string(index=False))

if __name__ == "__main__":
    solve_problem3()
