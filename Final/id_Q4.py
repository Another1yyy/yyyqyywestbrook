# -*- coding: utf-8 -*-
"""
Q4 · 圆柱目标遮蔽时间（单枚导弹，多云团并集）— 增强版：逐弹导出到 result2.xlsx 模板
--------------------------------------------------------------------------------
保留原判定与并集口径：ANY_POINT / FULL_ALL_POINT
新增：
- 在构建云团时记录投放点 R、起爆点 E、起爆时刻 t_e、UAV 名称/速度/方向
- 为“每一枚云团（烟幕弹）”单独计算其对 M1 的自身遮蔽时长（与全局并集无冲突）
- 导出到与 /mnt/data/result2.xlsx（Sheet1）一致的列名/顺序（每弹一行）

列名（严格对齐模板）：
['无人机编号','无人机运动方向','无人机运动速度 (m/s)',
 '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
 '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
 '有效干扰时长 (s)']
"""

import math
import numpy as np
import pandas as pd

# =========================
# 通用工具
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

def find_cover_intervals(f, t0, t1, dt=0.02):
    """在 [t0, t1] 上以步长 dt 扫描 f(t)（<=0 视为遮蔽），并用二分细化交界点。"""
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t); vs.append(f(t))
        t += dt

    roots = []
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        fa, fb = vs[i - 1], vs[i]
        if fa == 0.0: roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None: roots.append(r)
    roots = sorted(roots)

    def inside(tt): return f(tt) <= 0.0

    intervals, cur_in, cursor = [], inside(t0), t0
    for r in roots:
        if cur_in:
            intervals.append((cursor, r))
            cur_in = False
        else:
            cursor = r
            cur_in = True
    if cur_in:
        intervals.append((cursor, t1))
    return [(a, b) for (a, b) in intervals if b > a + 1e-8]

# =========================
# 圆柱体目标采样
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0      # 如题面不同请修改（你曾强调半径=100）
TARGET_H = 10.0     # 如题面不同请修改

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
    thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
    rs = np.linspace(0.0, radius, n_r_disk+1)[1:]
    for z in [0.0, height]:
        for r in rs:
            for th in thetas_d:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0] + r*c, base[1] + r*s, base[2] + z])
    return np.asarray(pts, dtype=float)

CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)

def dist_point_to_segments_batch(P, Ms, Xs):
    """单点 P 到一批线段 [M_i, X_i] 的距离（向量化）。"""
    BA = Xs - Ms
    BA2 = np.einsum('ij,ij->i', BA, BA)
    zero = BA2 < 1e-12
    BA2 = np.where(zero, 1.0, BA2)
    PA = P[None, :] - Ms
    s = np.einsum('ij,ij->i', PA, BA) / BA2
    s = np.clip(s, 0.0, 1.0)
    Q = Ms + s[:, None] * BA
    d = np.linalg.norm(P[None, :] - Q, axis=1)
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1); s[zero] = 0.0
    return d, s

# =========================
# 物理参数
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0  # 云团寿命
EVAL_DT = 0.02         # 评估步长

# 导弹（直指原点 O）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t): return M0 + v_m * t * u_m
T_hit = np.linalg.norm(M0) / v_m

# =========================
# 无人机/云团
# =========================
FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([17800.0,    0.0, 1800.0])
FY3_F0 = np.array([17800.0,    0.0, 1800.0])

def build_clouds(uav_list, skip_ground_explosion=True):
    """
    输入 UAV 参数列表，输出云团列表：
      cloud = {'E','R','t_e','uav','v','theta_deg'}
    """
    clouds = []
    for u in uav_list:
        name = u.get("name", "FY?")
        F0 = np.array(u.get("F0",
                            FY1_F0 if name == "FY1" else
                            FY2_F0 if name == "FY2" else
                            FY3_F0), dtype=float)
        v_u = float(u["v"])
        theta_deg = float(u["theta_deg"])
        theta = math.radians(theta_deg)
        h = np.array([math.cos(theta), math.sin(theta), 0.0])
        t_drop = u["t_drop"]; tau = u["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), f"{name}: t_drop 和 tau 长度不一致"

        for td, ta in zip(t_drop, tau):
            R = F0 + v_u * td * h
            E = R + v_u * ta * h + 0.5 * np.array([0.0, 0.0, -g]) * (ta**2)
            t_e = td + ta
            if skip_ground_explosion and E[2] <= 0.0:
                continue
            clouds.append({
                "E": E, "R": R, "t_e": t_e,
                "uav": name, "v": v_u, "theta_deg": theta_deg
            })
    return clouds

# =========================
# 遮蔽函数构造：ANY_POINT / FULL_ALL_POINT
# =========================
def make_f_any_full(clouds, cyl_pts=CYL_PTS):
    Xi = cyl_pts
    def f_any_point(t):
        M = missile_pos(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)
        best = np.inf; alive = False
        for c in clouds:
            te = c["t_e"]
            if (t < te) or (t > min(te + effective_span, T_hit)): continue
            alive = True
            C = c["E"] + np.array([0.0, 0.0, -sink_v]) * (t - te)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            best = min(best, float(np.min(d)))
        if not alive: return +1e9
        return best - R_cloud
    def f_full_all_point(t):
        M = missile_pos(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)
        any_alive = False
        dmin_over_clouds = np.full(len(Xi), np.inf, dtype=float)
        for c in clouds:
            te = c["t_e"]
            if (t < te) or (t > min(te + effective_span, T_hit)): continue
            any_alive = True
            C = c["E"] + np.array([0.0, 0.0, -sink_v]) * (t - te)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            dmin_over_clouds = np.minimum(dmin_over_clouds, d)
        if not any_alive: return +1e9
        worst_over_points = float(np.max(dmin_over_clouds))
        return worst_over_points - R_cloud
    return f_any_point, f_full_all_point

# =========================
# 单弹“自身时长”评估（与全局并集相互独立）
# =========================
def duration_for_single_cloud(cloud, mode='ANY_POINT', dt=EVAL_DT, cyl_pts=CYL_PTS):
    """
    对单一云团计算其自身遮蔽时长（仅该弹参与遮蔽）。
    mode: 'ANY_POINT' 或 'FULL_ALL_POINT'
    """
    f_any, f_full = make_f_any_full([cloud], cyl_pts)
    f = f_any if mode == 'ANY_POINT' else f_full
    t0 = cloud["t_e"]
    t1 = min(T_hit, cloud["t_e"] + effective_span)
    ivs = find_cover_intervals(f, t0, t1, dt=dt)
    return sum(b - a for a, b in ivs), ivs

# =========================
# 导出到 Excel（模板列名与顺序）
# =========================
TEMPLATE_COLUMNS_Q4 = [
    '无人机编号',
    '无人机运动方向',
    '无人机运动速度 (m/s)',
    '烟幕干扰弹投放点的x坐标 (m)',
    '烟幕干扰弹投放点的y坐标 (m)',
    '烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)',
    '烟幕干扰弹起爆点的y坐标 (m)',
    '烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)',
]

def export_to_excel_q4(clouds, out_path="result2_filled.xlsx",
                       single_mode='ANY_POINT', dt=EVAL_DT):
    """
    clouds: build_clouds(...) 的输出；逐弹评估自身时长并导出模板格式。
    single_mode: 计算单弹时长的口径（'ANY_POINT'/'FULL_ALL_POINT'）
    """
    rows = []
    for c in clouds:
        dur, _ = duration_for_single_cloud(c, mode=single_mode, dt=dt, cyl_pts=CYL_PTS)
        R = c["R"]; E = c["E"]
        rows.append({
            '无人机编号': c["uav"],
            '无人机运动方向': float(c["theta_deg"]),
            '无人机运动速度 (m/s)': float(c["v"]),
            '烟幕干扰弹投放点的x坐标 (m)': float(R[0]),
            '烟幕干扰弹投放点的y坐标 (m)': float(R[1]),
            '烟幕干扰弹投放点的z坐标 (m)': float(R[2]),
            '烟幕干扰弹起爆点的x坐标 (m)': float(E[0]),
            '烟幕干扰弹起爆点的y坐标 (m)': float(E[1]),
            '烟幕干扰弹起爆点的z坐标 (m)': float(E[2]),
            '有效干扰时长 (s)': float(dur),
        })
    df = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS_Q4)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    print(f"[OK] 已导出 -> {out_path}")

# =========================
# 主流程（保留全局并集评估以便核查）
# =========================
if __name__ == "__main__":
    # 示例 UAV 输入（可替换；多枚弹一弹一行）
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=111.431672, theta_deg=7.365624,
    #          t_drop=[0.0], tau=[0.8347]),
    #     dict(name="FY2", F0=FY2_F0, v=140, theta_deg=295.515,
    #          t_drop=[5.774535], tau=[5.007005]),
    #     dict(name="FY3", F0=FY3_F0, v=110, theta_deg=90,
    #          t_drop=[24], tau=[4]),
    # ]  # TIME=11.22
    UAVS = [
        dict(name="FY1", F0=FY1_F0, v=140, theta_deg=180,
             t_drop=[0], tau=[0]),
        dict(name="FY2", F0=FY2_F0, v=140, theta_deg=180,
             t_drop=[1.5891357], tau=[4.504]),
        dict(name="FY3", F0=FY3_F0, v=140, theta_deg=180,
             t_drop=[4.49948], tau=[5.73]),
    ]

    # 构建云团（记录 R/E/t_e/UAV 信息）
    clouds = build_clouds(UAVS, skip_ground_explosion=True)
    if not clouds:
        raise RuntimeError("没有有效云团（起爆高度<=0？请检查参数）。")

    # —— 全局并集（用于核对，与导出无冲突）
    f_any, f_full = make_f_any_full(clouds, CYL_PTS)
    OCCLUSION_MODE = 'FULL_ALL_POINT'  # 'ANY_POINT' or 'FULL_ALL_POINT'
    f = f_any if OCCLUSION_MODE == 'ANY_POINT' else f_full
    t0 = min(c["t_e"] for c in clouds)
    t1 = min(T_hit, max(c["t_e"] + effective_span for c in clouds))
    intervals = find_cover_intervals(f, t0, t1, dt=EVAL_DT)
    total = sum(b - a for a, b in intervals)

    print(f"=== 第四问 · 圆柱目标 · {OCCLUSION_MODE} ===")
    print(f"评估窗口: [{t0:.6f}, {t1:.6f}] s；云团数量: {len(clouds)}；采样点数: {len(CYL_PTS)}")
    if intervals:
        for i, (a, b) in enumerate(intervals, 1):
            print(f"- 区间{i}: [{a:.6f}, {b:.6f}]  dur={b-a:.6f} s")
        print(f"===> 遮蔽总时长(全局并集): {total:.6f} s")
    else:
        print("===> 无遮蔽。")

    # —— 逐弹自身时长导出（模板同 result2.xlsx）
    #     single_mode 取 'ANY_POINT'（通常更贴近“该弹自身是否产生遮蔽”）
    export_to_excel_q4(clouds, out_path="result2_filled.xlsx",
                       single_mode='FULL_ALL_POINT', dt=EVAL_DT)
