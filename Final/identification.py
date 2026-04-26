# -*- coding: utf-8 -*-
"""
Verifier for Problem 4 (point target)
- Input: 3 UAVs' (v, heading, drop times, fuse delays), optional F0
- Output: union occlusion time vs missile M1 and the intervals
- Target is a POINT (not a volume)
Dependencies: numpy
"""

import math
import numpy as np

# =========================
# 全局常量（可按题面需要调整）
# =========================
g = 9.8
R_cloud = 10.0          # 云团有效半径 (m)
smoke_duration = 20.0   # 云团寿命 (s)
sink_v = 3.0            # 云团向下沉降速度 (m/s)
vm = 300.0              # 导弹速度 (m/s)
EVAL_DT = 0.02          # 时间采样步长（可调：大→快，小→准）

# 目标点（把目标看成一个点）
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)

# 导弹 M1：从 M0 直飞“假目标原点”
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
u_m = (np.array([0.0,0.0,0.0]) - M0) / np.linalg.norm(M0)  # 朝原点单位向量
T_HIT = np.linalg.norm(M0) / vm                            # 击中原点所需时间

# 无人机默认初始位置（若实际不同可改）
FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([12000.0,  1400.0, 1400.0])
FY3_F0 = np.array([6000.0, -3000.0, 700.0])

# =========================
# 工具函数
# =========================
def missile_pos_vec(t):
    """导弹在绝对时刻 t (标量或数组) 的位置（向量化）。"""
    t = np.asarray(t, dtype=float)
    return M0[None,:] + (vm * t[:,None]) * u_m[None,:]

def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a,b))
    return out

def total_length(intervals):
    return sum(b-a for a,b in intervals)

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
    E = R + v * tau * h + 0.5 * np.array([0.0,0.0,-g]) * (tau**2)
    return R, E, (t_drop + tau)

def occlusion_signal_point(E, t_e, dt=EVAL_DT):
    """
    针对“点目标”构造遮蔽信号：
    f(t) = dist(云团中心C(t), 线段[ M(t) → T_POINT ]) - R_cloud
    返回绝对时刻区间（D<=R_cloud）的列表。
    """
    # 云团有效到期：寿命 or 导弹抵达原点
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0:
        return []

    # 时间网格（相对起爆时刻）
    T = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, T)
    tabs  = t_e + tgrid

    # 向量化计算
    M = missile_pos_vec(tabs)                                   # (T,3)
    C = E[None,:] + np.c_[np.zeros_like(tgrid), np.zeros_like(tgrid), -sink_v * tgrid]  # (T,3)

    S  = T_POINT[None,:] - M                                    # (T,3)
    L2 = np.einsum('ij,ij->i', S, S)
    L2 = np.where(L2 > 1e-12, L2, 1e-12)                        # 避免除零
    CM = C - M
    s  = np.clip(np.einsum('ij,ij->i', CM, S) / L2, 0.0, 1.0)   # 投影系数
    Q  = M + s[:,None] * S
    d  = np.linalg.norm(C - Q, axis=1)                          # (T,)

    inside = d <= R_cloud
    if not np.any(inside):
        return []

    # 线性插值求过零边界
    intervals = []
    i = 0
    while i < T-1:
        if inside[i]:
            j = i + 1
            while j < T and inside[j]:
                j += 1
            # 进入边界
            t_in = tgrid[i]
            if i > 0 and not inside[i-1]:
                f1, f2 = d[i-1]-R_cloud, d[i]-R_cloud
                t_in = tgrid[i-1] + (tgrid[i]-tgrid[i-1]) * (abs(f1)/(abs(f1)+abs(f2)))
            # 离开边界
            t_out = tgrid[j-1]
            if j < T and not inside[j]:
                f1, f2 = d[j-1]-R_cloud, d[j]-R_cloud
                t_out = tgrid[j-1] + (tgrid[j]-tgrid[j-1]) * (abs(f1)/(abs(f1)+abs(f2)))
            intervals.append((t_e + t_in, t_e + t_out))
            i = j
        else:
            i += 1
    return merge_intervals(intervals)

# =========================
# 主验证函数
# =========================
def verify_cover_time_point_target(
    uav_list,
    dt=EVAL_DT,
    print_details=True
):
    """
    uav_list: 长度为 3 的列表，每个元素是字典：
        {
            "name": "FY1",
            "F0": np.array([x0,y0,z0]),  # 可省略，用默认
            "v":  110.0,                 # m/s
            "theta_deg": 30.0,           # 航向角（度）
            "t_drop": [10.0, 18.0],      # 可为标量或列表
            "tau":    [2.0,  2.5],       # 与 t_drop 等长；也可为标量（自动广播）
        }
    返回： (total_time, union_intervals, per_bomb_intervals)
    """
    # 默认初始点
    default_F0 = {"FY1": FY1_F0, "FY2": FY2_F0, "FY3": FY3_F0}

    all_intervals = []
    per_bomb = []  # [(uav_name, idx, tin, tout, dur, Ez, te), ...]

    for uav in uav_list:
        name = uav.get("name", "FY1")
        F0   = np.array(uav.get("F0", default_F0.get(name, FY1_F0)), dtype=float)
        v    = float(uav["v"])
        th   = math.radians(float(uav["theta_deg"]))

        t_drop = uav["t_drop"]
        tau    = uav["tau"]
        # 广播到列表
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), "t_drop 与 tau 长度不一致"

        for k,(td, ta) in enumerate(zip(t_drop, tau), start=1):
            R, E, t_e = explosion_point(F0, v, th, float(td), float(ta))
            if E[2] <= 0.0:
                if print_details:
                    print(f"[WARN] {name}#{k} 起爆高度 E_z={E[2]:.3f} <= 0，跳过该弹")
                continue
            ivs = occlusion_signal_point(E, t_e, dt=dt)
            all_intervals += ivs
            dur = total_length(ivs)
            per_bomb.append((name, k, ivs[0][0], ivs[-1][1], dur, E[2], t_e) if ivs else (name, k, np.nan, np.nan, 0.0, E[2], t_e))

    union_ivs = merge_intervals(all_intervals)
    total = total_length(union_ivs)

    if print_details:
        print("\n=== 验证结果（目标当作点） ===")
        print(f"- 并集遮蔽总时长: {total:.6f} s")
        if union_ivs:
            for i,(a,b) in enumerate(union_ivs, 1):
                print(f"  · 并集区间{i}: [{a:.6f}, {b:.6f}] s，时长 {(b-a):.6f} s")
        else:
            print("  · 无遮蔽区间")
        for (name, k, tin, tout, dur, Ez, te) in per_bomb:
            print(f"- {name}#{k}: 起爆时刻 t_e={te:.6f} s, 起爆高度 E_z={Ez:.3f} m, 自身遮蔽时长={dur:.6f} s"
                  + ("" if np.isnan(tin) else f"，首段≈[{tin:.6f},{tout:.6f}]"))

    return total, union_ivs, per_bomb

# =========================
# 示例：在这里填入三架无人机参数
# =========================
if __name__ == "__main__":
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=125.0011, theta_deg=-169.02,
    #          t_drop=[46.2704], tau=[11.5499]),
    #     dict(name="FY2", F0=FY2_F0, v=114.0954, theta_deg=-144.962,
    #          t_drop=[37.5771], tau=[9.3023]),
    #     dict(name="FY3", F0=FY3_F0, v=139.1021, theta_deg=120.9935,
    #          t_drop=[18.6733], tau=[7.548]),
    # ]
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=70, theta_deg=7.938,
    #          t_drop=[0.1885], tau=[0.97]),
    #     dict(name="FY2", F0=FY2_F0, v=115.2049, theta_deg=-98.7115,
    #          t_drop=[5.5214], tau=[6.4082]),
    #     dict(name="FY3", F0=FY3_F0, v=104.2858, theta_deg=139.7683,
    #          t_drop=[36.4923], tau=[9.6639]),
    # ]
    # TIME = 12.6s
    # UAVS = [
    #     dict(name="FY1", F0=FY1_F0, v=70, theta_deg=7.938,
    #          t_drop=[0.1885], tau=[0.97]),
    #     dict(name="FY2", F0=FY2_F0, v=121.5849, theta_deg=-113.939,
    #          t_drop=[5.3939], tau=[6.8988]),
    #     dict(name="FY3", F0=FY3_F0, v=136.0, theta_deg=129.6203,
    #          t_drop=[21.614], tau=[8.1093]),
    # ]  #TIME=13.214558
    UAVS = [
        dict(name="FY1", F0=FY1_F0, v=111.431672, theta_deg=7.365624,
             t_drop=[0.0], tau=[0.8347]),
        dict(name="FY2", F0=FY2_F0, v=139.030360, theta_deg=-104.501994,
             t_drop=[3.1721702825], tau=[6.80987581111352]),
        dict(name="FY3", F0=FY3_F0, v=139.857096, theta_deg=131.046332,
             t_drop=[21.106055753098158], tau=[8.624888988896926]),
    ]  #TIME=15.482417
    verify_cover_time_point_target(UAVS, dt=EVAL_DT, print_details=True)
