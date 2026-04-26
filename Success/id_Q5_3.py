# -*- coding: utf-8 -*-
"""
Per-Bomb Interference Reporter (Point Target) — with Release Point & Excel Export
---------------------------------------------------------------------------------
新增:
- 逐枚烟幕弹的投放点 R 与投放时刻 t_drop（已在上一版加入）
- 导出到与模板(result3.xlsx, Sheet1)相同列名/顺序的Excel

说明:
- "有效干扰时长 (s)" 默认为对各导弹遮蔽总时长之和(aggregate='sum')
  可切换为 'max' 取各导弹遮蔽时长的最大值
- "干扰的导弹编号" 使用 ';' 拼接, 如 "M1;M3"
"""

import math
import numpy as np
import pandas as pd
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
    输出结果records字段新增 'v' 与 'theta_deg' 便于导出。
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
        th_deg = float(uav["theta_deg"])
        th = math.radians(th_deg)
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
                "v": v,                   # 新增: 速度
                "theta_deg": th_deg,      # 新增: 方向(度)
                "bomb_idx": k,
                "R": (float(R[0]), float(R[1]), float(R[2])),
                "t_drop": float(td),
                "E": (float(E[0]), float(E[1]), float(E[2])),
                "t_e": float(t_e),
                "per_missile": per_m,
                "interfered_missiles": interfered
            }
            results.append(rec)

            if verbose:
                print(f"\n[{name}#{k}]")
                print(f"  R(投放)=({R[0]:.3f},{R[1]:.3f},{R[2]:.3f}), t_drop={td:.3f}s")
                print(f"  E(起爆)=({E[0]:.3f},{E[1]:.3f},{E[2]:.3f}), t_e={t_e:.3f}s")
                for m_name in per_m:
                    dur = per_m[m_name]["duration"]
                    print(f"  - {m_name}: dur={dur:.6f}s, intervals={per_m[m_name]['intervals']}")
                print(f"  -> 干扰到: {interfered if interfered else '无'}")

    return results

# =========================
# 导出: 与模板(result3.xlsx, Sheet1)一致列名/顺序
# =========================
TEMPLATE_COLUMNS = [
    '无人机编号',
    '无人机运动方向',
    '无人机运动速度 (m/s)',
    '烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)',
    '烟幕干扰弹投放点的y坐标 (m)',
    '烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)',
    '烟幕干扰弹起爆点的y坐标 (m)',
    '烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)',
    '干扰的导弹编号'
]

def export_to_excel(results: List[Dict[str, Any]],
                    out_path: str = "result5.xlsx",
                    aggregate: str = 'sum'):
    """
    将 report_per_bomb_interference 的 results 导出为 Excel (Sheet1)，表头对齐模板。
    aggregate: 'sum' -> 各导弹遮蔽时长求和; 'max' -> 取最大值
    """
    rows = []
    for rec in results:
        uav = rec['uav']
        v = rec['v']
        theta_deg = rec['theta_deg']
        idx = rec['bomb_idx']
        Rx, Ry, Rz = rec['R']
        Ex, Ey, Ez = rec['E']
        per_m = rec['per_missile']
        ids = [mn for mn, info in per_m.items() if info['duration'] > 1e-9]
        if aggregate == 'sum':
            eff = float(sum(info['duration'] for info in per_m.values()))
        elif aggregate == 'max':
            eff = float(max((info['duration'] for info in per_m.values()), default=0.0))
        else:
            raise ValueError("aggregate 必须是 'sum' 或 'max'")

        rows.append({
            '无人机编号': uav,
            '无人机运动方向': float(theta_deg),           # 以“度”数值输出；如需字符串可改为 f"{theta_deg:.3f}°"
            '无人机运动速度 (m/s)': float(v),
            '烟幕干扰弹编号': int(idx),
            '烟幕干扰弹投放点的x坐标 (m)': float(Rx),
            '烟幕干扰弹投放点的y坐标 (m)': float(Ry),
            '烟幕干扰弹投放点的z坐标 (m)': float(Rz),
            '烟幕干扰弹起爆点的x坐标 (m)': float(Ex),
            '烟幕干扰弹起爆点的y坐标 (m)': float(Ey),
            '烟幕干扰弹起爆点的z坐标 (m)': float(Ez),
            '有效干扰时长 (s)': eff,
            '干扰的导弹编号': ';'.join(ids) if ids else ''
        })

    df = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    print(f"[OK] 导出成功 -> {out_path}")

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
    # UAVS = [
    #     dict(name="FY1", v=140.0, theta_deg=6.744,   t_drop=[0.000, 45.039, 66.999], tau=[0.732, 14.999, 0.000]),
    #     dict(name="FY2", v=98.227153, theta_deg=282.483654, t_drop=[7.198853514385463, 8.198854514385461, 49.992508461305576], tau=[2.9630817130408045, 2.192383209103564, 5.03745550276874]),
    #     dict(name="FY3", v=140.0,  theta_deg=82.250, t_drop=[18.714, 19.714, 22.423], tau=[1.982, 2.490, 11.951]),
    #     dict(name="FY4", v=125.811,theta_deg=-90.036,t_drop=[1.899, 3.608, 41.055],  tau=[10.855, 11.893, 4.491]),
    #     dict(name="FY5", v=111.065118,   theta_deg=122.540507, t_drop=[0.0, 14.533026756493195, 55.59243510604091], tau=[7.333341859137445, 2.6121651315338434, 15.0]),
    # ]
    UAVS = [
        dict(name="FY1", v=140.0, theta_deg=180, t_drop=[0.000, 1.5891357, 4.49948], tau=[0, 4.504, 5.73]),
        dict(name="FY2", v=98.227153, theta_deg=282.483654,
             t_drop=[7.198853514385463, 8.198854514385461, 49.992508461305576],
             tau=[2.9630817130408045, 2.192383209103564, 5.03745550276874]),
        dict(name="FY3", v=140.0, theta_deg=82.250, t_drop=[18.714, 19.714, 22.423], tau=[1.982, 2.490, 11.951]),
        dict(name="FY4", v=125.811, theta_deg=-90.036, t_drop=[1.899, 3.608, 41.055], tau=[10.855, 11.893, 4.491]),
        dict(name="FY5", v=111.065118, theta_deg=122.540507, t_drop=[0.0, 14.533026756493195, 55.59243510604091],
             tau=[7.333341859137445, 2.6121651315338434, 15.0]),
    ]

    results = report_per_bomb_interference(MISSILES, UAVS, dt=EVAL_DT, verbose=True)
    # 导出 (你可以改 out_path 到你想要的位置；aggregate 可选 'sum' 或 'max')
    export_to_excel(results, out_path="result5.xlsx", aggregate='sum')
