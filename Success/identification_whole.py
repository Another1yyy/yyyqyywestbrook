# -*- coding: utf-8 -*-
"""
Q4 计算：圆柱目标遮蔽时间（单枚导弹，多云团并集）
- 物理设定沿用前述：云团半径 10 m、寿命 20 s、起爆后以 3 m/s 下沉；
  导弹从 M0 以 300 m/s 直线飞向原点。
- 目标：圆柱，默认圆心 (0,200,0)，半径 TARGET_R，高度 TARGET_H（按题面修改）。
- 遮蔽判定提供两种口径：
    1) ANY_POINT：       ∃P∈Cylinder, ∃cloud  s.t. dist(C_cloud(t), segment[M(t),P]) ≤ R_cloud
    2) FULL_ALL_POINT：  ∀P∈Cylinder, ∃cloud  s.t. dist(C_cloud(t), segment[M(t),P]) ≤ R_cloud
- 在 [min t_e, min(T_hit, max t_e+20)] 上用“采样+扫描+二分”求满足条件的所有区间；并集区间总时长即为遮蔽时间。
"""

import math
import numpy as np

# =========================
# 工具（与你提供的脚本一致的风格）
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
    """
    在 [t0, t1] 上以步长 dt 扫描 f(t)（<=0 视为“遮蔽”），并用二分细化交界点。
    返回所有区间 [(t_in, t_out), ...]
    """
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t)
        vs.append(f(t))
        t += dt

    roots = []
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        fa, fb = vs[i - 1], vs[i]
        if fa == 0.0:
            roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None:
                roots.append(r)
    roots = sorted(roots)

    def inside(t): return f(t) <= 0.0

    intervals = []
    cur_in = inside(t0)
    cursor = t0
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
# 圆柱体目标采样（按需调密）
# =========================
TARGET_BASE = np.array([0.0, 200.0, 0.0])  # 下底圆心
TARGET_R = 7.0   # ← 若题面为其它值请改（你曾强调半径=100）
TARGET_H = 10.0    # ← 按题面修改

def sample_cylinder_points(base, radius, height,
                           n_theta_side=64, n_z_side=8,
                           n_theta_disk=48, n_r_disk=3):
    """
    生成圆柱体表面采样点：
    - 侧面：n_theta_side × (n_z_side+1)
    - 上/下底面：极坐标网格（不含 r=0，只取同心环）
    返回 ndarray 形状 (N, 3)
    """
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

    # 上/下底：极坐标环
    thetas_d = np.linspace(0, 2*np.pi, n_theta_disk, endpoint=False)
    rs = np.linspace(0.0, radius, n_r_disk+1)[1:]  # 去掉 r=0
    for z in [0.0, height]:
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
    BA2 = np.where(zero, 1.0, BA2)  # 避免除0

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
# 物理参数
# =========================
g = 9.8
v_m = 300.0
R_cloud = 10.0
sink_v = 3.0
effective_span = 20.0  # 云团寿命

# 导弹（直指假目标原点）
M0 = np.array([20000.0, 0.0, 2000.0])
u_m = unit(-M0)
def missile_pos(t):
    return M0 + v_m * t * u_m

T_hit = np.linalg.norm(M0) / v_m  # 击中原点时刻（用于裁剪窗口）

# =========================
# 无人机/云团定义（可放 3~5 架，每架 ≤3 枚）
# =========================
# 默认初始位置（题面）
FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([12000.0,  1400.0, 1400.0])
FY3_F0 = np.array([ 6000.0, -3000.0,  700.0])

def build_clouds(uav_list):
    """
    输入 UAV 参数列表，输出云团列表：
      cloud = {'E': 起爆点, 't_e': 起爆时刻}
    """
    clouds = []
    for u in uav_list:
        name = u.get("name", "FY?")
        F0 = np.array(u.get("F0", FY1_F0 if name == "FY1" else FY2_F0 if name=="FY2" else FY3_F0), dtype=float)
        v_u = float(u["v"])
        theta = math.radians(float(u["theta_deg"]))
        h = np.array([math.cos(theta), math.sin(theta), 0.0])
        t_drop = u["t_drop"]; tau = u["tau"]
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), f"{name}: t_drop 和 tau 长度不一致"
        for td, ta in zip(t_drop, tau):
            # 投放点/起爆点
            R = F0 + v_u * td * h
            E = R + v_u * ta * h + 0.5 * np.array([0.0, 0.0, -g]) * (ta**2)
            t_e = td + ta
            if E[2] > 0.0:
                clouds.append({"E": E, "t_e": t_e})
    return clouds

# =========================
# 多云团口径：ANY_POINT / FULL_ALL_POINT
# =========================
def make_f_any_full(clouds, cyl_pts=CYL_PTS):
    Xi = cyl_pts  # (N,3)

    def f_any_point(t):
        """
        存在式：∃P, ∃cloud 使得被遮蔽。
        等价于：min_cloud min_P d(C_cloud,P; M(t)) - R_cloud
        无活跃云团时返回 +inf（不遮蔽）。
        """
        M = missile_pos(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)

        best = np.inf
        alive = False
        for c in clouds:
            t_e = c["t_e"]
            if (t < t_e) or (t > min(t_e + effective_span, T_hit)):
                continue
            alive = True
            C = c["E"] + np.array([0.0, 0.0, -sink_v]) * (t - t_e)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            best = min(best, float(np.min(d)))
        if not alive:
            return +1e9
        return best - R_cloud

    def f_full_all_point(t):
        """
        全遮蔽：∀P, ∃cloud 使得被遮蔽。
        等价于：max_P min_cloud d(C_cloud,P; M(t)) - R_cloud
        无活跃云团时返回 +inf（不遮蔽）。
        """
        M = missile_pos(t)
        Ms = np.repeat(M[None, :], len(Xi), axis=0)

        # 先对每个 P 计算“对云团的最小距离”，再在 P 上取最大。
        # 若在该 t 没有云团活跃，返回 +inf。
        any_alive = False
        dmin_over_clouds = np.full(len(Xi), np.inf, dtype=float)

        for c in clouds:
            t_e = c["t_e"]
            if (t < t_e) or (t > min(t_e + effective_span, T_hit)):
                continue
            any_alive = True
            C = c["E"] + np.array([0.0, 0.0, -sink_v]) * (t - t_e)
            d, _ = dist_point_to_segments_batch(C, Ms, Xi)
            dmin_over_clouds = np.minimum(dmin_over_clouds, d)

        if not any_alive:
            return +1e9

        worst_over_points = float(np.max(dmin_over_clouds))
        return worst_over_points - R_cloud

    return f_any_point, f_full_all_point

# =========================
# 主流程
# =========================
if __name__ == "__main__":
    # —— 示例 UAV 输入：可按需要增删、每架 ≤3 枚
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=111.431672, theta_deg=7.365624,
    #          t_drop=[0.0], tau=[0.8347]),
    #     dict(name="FY2", F0=FY2_F0, v=139.030360, theta_deg=-104.501994,
    #          t_drop=[3.1721702825], tau=[6.8098758111]),
    #     dict(name="FY3", F0=FY3_F0, v=139.857096, theta_deg=131.046332,
    #          t_drop=[21.1060557531], tau=[8.6248889889]),
    # ]
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=76.706, theta_deg=7.356,
    #          t_drop=[0.202130], tau=[0.871809]),
    #     dict(name="FY2", F0=FY2_F0, v=140, theta_deg=295.515,
    #          t_drop=[5.774535], tau=[5.007005]),
    #     dict(name="FY3", F0=FY3_F0, v=110, theta_deg=-180,
    #          t_drop=[4], tau=[24]),
    # ]
    UAVS = [
        dict(name="FY1", F0=FY1_F0, v=120, theta_deg=180,
             t_drop=[1.5], tau=[3.6])
    ]

    clouds = build_clouds(UAVS)
    if not clouds:
        raise RuntimeError("没有有效云团（起爆高度<=0？请检查参数）。")

    f_any, f_full = make_f_any_full(clouds, CYL_PTS)

    # —— 口径选择：
    OCCLUSION_MODE = 'FULL_ALL_POINT'   # 'ANY_POINT' 或 'FULL_ALL_POINT'
    f = f_any if OCCLUSION_MODE == 'ANY_POINT' else f_full

    # —— 评估时间窗：从最早起爆到导弹命中或最后云团消亡
    t0 = min(c["t_e"] for c in clouds)
    t1 = min(T_hit, max(c["t_e"] + effective_span for c in clouds))

    intervals = find_cover_intervals(f, t0, t1, dt=0.02)
    total = sum(b - a for a, b in intervals)

    print(f"=== 第四问 · 圆柱目标 · {OCCLUSION_MODE} ===")
    print(f"评估窗口: [{t0:.6f}, {t1:.6f}] s；云团数量: {len(clouds)}；采样点数: {len(CYL_PTS)}")
    if intervals:
        for i, (a, b) in enumerate(intervals, 1):
            print(f"- 区间{i}: [{a:.6f}, {b:.6f}]  dur={b-a:.6f} s")
        print(f"===> 遮蔽总时长: {total:.6f} s")
    else:
        print("===> 无遮蔽。")
