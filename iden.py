#!/usr/bin/env python3
# validate_q4.py
import numpy as np
import sys

# ---------------- 给定常量 ----------------
# 无人机初始位置
POS_FY = np.array([
    [17800, 0,   1800],   # FY1
    [12000, 1400, 1400],  # FY2
    [6000, -3000, 700]    # FY3
])
# 真目标
TARGET = np.array([0, 200, 0])
# 烟幕参数
R_SMOKE = 10.0          # 有效半径 m
V_SINK  = 3.0           # 下沉速度 m/s
T_ARM   = 3.6           # 从投弹到起爆 s
# 导弹M1匀速直线模型（用于计算“弹目连线”时刻 t_missile）
M1_POS0 = np.array([20000, 0, 2000])
M1_V    = np.array([-300, 0, 0])   # 300 m/s 沿 -x
# 仿真时间区间 [0, T_MAX] s
T_MAX = 80.0
DT    = 0.01

# ---------------- 工具函数 ----------------
def spherical2vec(v, theta_deg):
    """把 v(m/s) + theta(°) 转成二维速度向量 [vx, vy]"""
    theta = np.deg2rad(theta_deg)
    return np.array([v*np.cos(theta), v*np.sin(theta)])

def smoke_center(p_drop, t_from_burst):
    """起爆后 t_from_burst 秒时的烟幕球心"""
    # 起爆瞬间球心 = 投放点 + 3.6 s 自由下落 (仅 z 方向)
    # 以后匀速下沉
    z_burst = p_drop[2] - 0.5*9.8*T_ARM**2
    return np.array([
        p_drop[0],
        p_drop[1],
        z_burst - V_SINK*t_from_burst
    ])

# ---------------- 主计算 ----------------
def calc_coverage(params):
    """
    params: 3×3 数组/列表
        [[v1, theta1, t_drop1],
         [v2, theta2, t_drop2],
         [v3, theta3, t_drop3]]
    返回总遮蔽时长（秒）
    """
    covered = np.zeros(int(np.ceil(T_MAX/DT)), dtype=bool)
    for i in range(3):
        v, theta, t_drop = params[i]
        vxy = spherical2vec(v, theta)
        # 投放点 = 初始位置 + 匀速直线飞行 t_drop 秒
        p_drop = POS_FY[i][:2] + vxy*t_drop
        p_drop = np.array([p_drop[0], p_drop[1], POS_FY[i][2]])
        # 起爆时刻
        t_burst = t_drop + T_ARM
        # 在仿真时间轴上标记遮蔽区间
        for k, t in enumerate(np.arange(0, T_MAX, DT)):
            if t < t_burst:
                continue
            center = smoke_center(p_drop, t - t_burst)
            if np.linalg.norm(center - TARGET) <= R_SMOKE:
                covered[k] = True
    # 累计时长
    return covered.sum() * DT

# ---------------- 读入 & 输出 ----------------
def main():
    if len(sys.argv) == 1:
        # 示例：直接写死三组参数
        params = np.array([
            [120,  45, 5.0],
            [100, -30, 6.0],
            [140,  90, 7.0]
        ])
    else:
        # 命令行读 9 个数字
        if len(sys.argv) != 10:
            print("用法: python validate_q4.py v1 theta1 t_drop1 v2 theta2 t_drop2 v3 theta3 t_drop3")
            sys.exit(1)
        nums = list(map(float, sys.argv[1:10]))
        params = np.array(nums).reshape(3, 3)

    total = calc_coverage(params)
    print(f"total_cover = {total:.3f} s")

if __name__ == "__main__":
    main()