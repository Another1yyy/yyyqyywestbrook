# -*- coding: utf-8 -*-
"""
Per-Bomb Interference Reporter (Point Target) — with Release Point
------------------------------------------------------------------
新增:
- 逐枚烟幕弹输出投放坐标 R=(Rx,Ry,Rz) 与投放时刻 t_drop

功能:
- 输入多架无人机（每架最多3枚）参数，逐枚烟幕弹计算:
  * 投放点 R 与投放时刻 t_drop
  * 起爆点 E 与起爆时刻 t_e = t_drop + tau
  * 对每枚导弹的遮蔽区间并集与总时长
  * 被干扰到的导弹编号列表

建模/物理假设:
- 点目标 T = (0, 200, 0)
- 云团: 有效半径 10 m; 自起爆后匀速下沉 3 m/s; 有效寿命 20 s
- 导弹: 速率 300 m/s, 直线飞向原点 O=(0,0,0)
- 遮蔽判据: 距离(云团中心, 线段[ M(t) -> T ]) <= 10 m

作者: 为 Westbrook 定制
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any

# =========================
# 全局常量 (可按需调整)
# =========================
g = 9.8
R_cloud = 10.0        # 云团有效半径 [m]
smoke_duration = 20.0 # 云团有效寿命 [s]
sink_v = 3.0          # 云团中心下沉速度 [m/s]
vm = 300.0            # 导弹速度 [m/s]
EVAL_DT = 0.02        # 仿真步长

# 目标(点)
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)

# 导弹初始位置 (可覆盖)
M1_M0 = np.array([20000.0,     0.0, 2000.0])
M2_M0 = np.array([19000.0,   600.0, 2100.0])
M3_M0 = np.array([18000.0,  -600.0, 1900.0])

# 无人机初始位置 (可覆盖)
FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([12000.0,  1400.0, 1400.0])
FY3_F0 = np.array([ 6000.0, -3000.0,  700.0])
FY4_F0 = np.array([11000.0,  2000.0, 1800.0])
FY5_F0 = np.array([13000.0, -2000.0, 1300.0])
DEFAULT_F0 = {"FY1": FY1_F0, "FY2": FY2_F0, "FY3": FY3_F0, "FY4": FY4_F0, "FY5": FY5_F0}

# =========================
# 区间工具
# =========================
def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def total_length(intervals: List[Tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in intervals)

# =========================
# 运动学
# =========================
def missile_dir_u_and_hit_time(M0: np.ndarray) -> Tuple[np.ndarray, float]:
    d = np.linalg.norm(M0)
    u = -M0 / d
    return u, d / vm

def missile_pos_vec(t_abs: np.ndarray, M0: np.ndarray, u_m: np.ndarray) -> np.ndarray:
    t_abs = np.asarray(t_abs, dtype=float)
    return M0[None, :] + (vm * t_abs[:, None]) * u_m[None, :]

def explosion_point(F0: np.ndarray, v: float, theta_rad: float, t_drop: float, tau: float):
    """
    无人机等高匀速直线: 先飞行 t_drop 再投放; 脱离后经历 tau 再起爆。
    返回:
      R: 投放点
      E: 起爆点
      t_e: 起爆时刻 (绝对时刻, 若 t=0 为起飞/参考时刻)
    """
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)

# =========================
# 单枚“云团 vs 单枚导弹”的遮蔽区间
# =========================
def occlusion_intervals_point_target_for_missile(
    E: np.ndarray, t_e: float, M0: np.ndarray, dt: float = EVAL_DT
) -> List[Tuple[float, float]]:
    u_m, T_HIT = missile_dir_u_and_hit_time(M0)
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0:
        return []

    T = int(math.floor(span / dt)) + 1
    tgrid_rel = np.linspace(0.0, span, T)            # 相对起爆时刻
    tgrid_abs = t_e + tgrid_rel                      # 绝对时刻

    M = missile_pos_vec(tgrid_abs, M0, u_m)          # (T,3)
    C = E[None, :] + np.c_[np.zeros_like(tgrid_rel),
                           np.zeros_like(tgrid_rel),
                           -sink_v * tgrid_rel]       # 云团中心

    S = T_POINT[None, :] - M                         # (T,3)
    L2 = np.einsum('ij,ij->i', S, S)
    L2 = np.where(L2 > 1e-12, L2, 1e-12)

    CM = C - M
    s = np.clip(np.einsum('ij,ij->i', CM, S) / L2, 0.0, 1.0)
    Q = M + s[:, None] * S
    d = np.linalg.norm(C - Q, axis=1)

    inside = d <= R_cloud
    if not np.any(inside):
        return []

    intervals = []
    i = 0
    while i < T - 1:
        if inside[i]:
            j = i + 1
            while j < T and inside[j]:
                j += 1
            # 线性内插入/出
            t_in = tgrid_rel[i]
            if i > 0 and not inside[i - 1]:
                f1, f2 = d[i - 1] - R_cloud, d[i] - R_cloud
                t_in = tgrid_rel[i - 1] + (tgrid_rel[i] - tgrid_rel[i - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            t_out = tgrid_rel[j - 1]
            if j < T and not inside[j]:
                f1, f2 = d[j - 1] - R_cloud, d[j] - R_cloud
                t_out = tgrid_rel[j - 1] + (tgrid_rel[j] - tgrid_rel[j - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            intervals.append((t_e + t_in, t_e + t_out))
            i = j
        else:
            i += 1

    return merge_intervals(intervals)

# =========================
# 主函数: 逐枚烟幕弹报告
# =========================
def report_per_bomb_interference(
    missiles: List[Dict[str, Any]],
    uav_list: List[Dict[str, Any]],
    dt: float = EVAL_DT,
    skip_ground_explosion: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    输入:
      missiles: [{'name': 'M1', 'M0': np.array([...])}, ...]
      uav_list: 每架UAV字典:
        {
          'name': 'FY1',
          'F0': np.array([x,y,z])  # 可省略, 有默认
          'v': 120.0,
          'theta_deg': 30.0,
          't_drop': [..] 或 标量,
          'tau':    [..] 或 标量 (可按t_drop广播)
        }

    输出:
      results: 列表, 每个元素是一枚烟幕弹的报告:
        {
          'uav': 'FY1',
          'bomb_idx': 1,
          'R': (Rx,Ry,Rz),          # 新增: 投放点
          't_drop': t_drop,         # 新增: 投放时刻
          'E': (Ex,Ey,Ez),
          't_e': t_e,
          'per_missile': {
              'M1': {'intervals': [(a,b), ...], 'duration': dur},
              'M2': {...},
              'M3': {...}
          },
          'interfered_missiles': ['M1','M3']
        }
    """
    # 归一化导弹
    missiles_norm = []
    for m in missiles:
        name = m.get("name", "M?")
        M0 = np.array(m["M0"], dtype=float)
        missiles_norm.append((name, M0))

    results = []

    for uav in uav_list:
        name = uav.get("name", "FY?")
        F0 = np.array(uav.get("F0", DEFAULT_F0.get(name, FY1_F0)), dtype=float)
        v = float(uav["v"])
        th = math.radians(float(uav["theta_deg"]))
        t_drop = uav["t_drop"]
        tau    = uav["tau"]

        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), f"{name}: t_drop 与 tau 长度不一致"

        for k, (td, ta) in enumerate(zip(t_drop, tau), start=1):
            R, E, t_e = explosion_point(F0, v, th, float(td), float(ta))

            if skip_ground_explosion and E[2] <= 0.0:
                if verbose:
                    print(f"[WARN] {name}#{k} 起爆高度 Ez={E[2]:.3f}<=0, 跳过。 (R=({R[0]:.3f},{R[1]:.3f},{R[2]:.3f}), t_drop={td:.3f})")
                continue

            per_m = {}
            for m_name, M0 in missiles_norm:
                ivs = occlusion_intervals_point_target_for_missile(E, t_e, M0, dt=dt)
                per_m[m_name] = {
                    "intervals": ivs,
                    "duration": total_length(ivs)
                }

            interfered = [mn for mn, rec in per_m.items() if rec["duration"] > 1e-9]

            rec = {
                "uav": name,
                "bomb_idx": k,
                "R": (float(R[0]), float(R[1]), float(R[2])),   # 新增
                "t_drop": float(td),                            # 新增
                "E": (float(E[0]), float(E[1]), float(E[2])),
                "t_e": float(t_e),
                "per_missile": per_m,
                "interfered_missiles": interfered
            }
            results.append(rec)

            if verbose:
                print(f"\n[{name}#{k}]")
                print(f"  R(投放)=({R[0]:.3f},{R[1]:.3f},{R[2]:.3f}), t_drop={td:.3f}s")  # 新增打印
                print(f"  E(起爆)=({E[0]:.3f},{E[1]:.3f},{E[2]:.3f}), t_e={t_e:.3f}s")
                for m_name in per_m:
                    dur = per_m[m_name]["duration"]
                    print(f"  - {m_name}: dur={dur:.6f}s, intervals={per_m[m_name]['intervals']}")
                print(f"  -> 干扰到: {interfered if interfered else '无'}")

    return results

# =========================
# 示例
# =========================
if __name__ == "__main__":
    # 导弹 (可覆盖)
    MISSILES = [
        dict(name="M1", M0=M1_M0),
        dict(name="M2", M0=M2_M0),
        dict(name="M3", M0=M3_M0),
    ]

    # UAV 输入示例（你可直接替换为自己的搜索/策略结果）
    UAVS = [
        dict(name="FY1", v=140.0, theta_deg=6.744,   t_drop=[0.000, 45.039, 66.999], tau=[0.732, 14.999, 0.000]),
        dict(name="FY2", v=122.76, theta_deg=43.369, t_drop=[0.000, 48.654, 66.999], tau=[9.960, 15.000, 15.000]),
        dict(name="FY3", v=140.0,  theta_deg=82.250, t_drop=[18.714, 19.714, 22.423], tau=[1.982, 2.490, 11.951]),
        dict(name="FY4", v=125.811,theta_deg=-90.036,t_drop=[1.899, 3.608, 41.055],  tau=[10.855, 11.893, 4.491]),
        dict(name="FY5", v=70.0,   theta_deg=-180.0, t_drop=[11.195, 21.095, 66.608], tau=[9.961, 10.846, 13.662]),
    ]

    _ = report_per_bomb_interference(MISSILES, UAVS, dt=EVAL_DT, verbose=True)
