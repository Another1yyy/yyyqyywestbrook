import numpy as np

# --- 1. 定义常量和初始状态 ---
# 导弹 M1
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M = 300.0

# 无人机 FY1
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
V_FY1 = 120.0

# 目标
P_TT_BASE = np.array([0.0, 200.0, 0.0])

# 烟幕云团
R_SC = 10.0
V_SC_DOWN = 3.0
SMOKE_DURATION = 20.0

# 物理和事件常量
G = 9.8  # 重力加速度
T_ASSIGN = 1.5  # 受领任务时刻
T_INTERVAL = 3.6  # 投放后到起爆的间隔


# --- 2. 建立运动学和几何函数 ---

def get_missile_pos(t):
    """计算 t 时刻导弹 M1 的位置"""
    # 导弹朝向假目标原点 (0,0,0) 飞行
    direction_m = -P_M1_0 / np.linalg.norm(P_M1_0)
    return P_M1_0 + V_M * t * direction_m

def get_uav_pos(t):
    """计算 t 时刻无人机 FY1 的位置"""
    # 无人机朝向假目标原点 (0,0,0) 飞行
    direction_fy = -P_FY1_0 / np.linalg.norm(P_FY1_0)
    if t < T_ASSIGN:
        return P_FY1_0
    # t >= T_ASSIGN, 无人机开始移动
    return P_FY1_0 + V_FY1 * (t - T_ASSIGN) * direction_fy


def get_bomb_pos(t, t_drop, p_drop, v_drop):
    """计算烟幕弹在 t 时刻的位置 (平抛运动)"""
    delta_t = t - t_drop
    # 水平位移
    pos_xy = p_drop[:2] + v_drop[:2] * delta_t
    # 垂直位移
    pos_z = p_drop[2] + v_drop[2] * delta_t - 0.5 * G * delta_t ** 2
    return np.array([pos_xy[0], pos_xy[1], pos_z])
### 袁浥原：计算explosion发生的位置
def calculate_explosion_pos(V_FY,T_INTERVAL,pos_start):
    pos_xyz = V_FY[:3] * T_INTERVAL + pos_start[:3]
    return pos_xyz
def calculate_smoke_pos(explosion_pos,t_drop):
    smoke_pos_z = explosion_pos[2] + 3.00 * t_drop
    smoke_pos_x = explosion_pos[0]
    smoke_pos_y = explosion_pos[1]
    smoke_pos = [smoke_pos_x, smoke_pos_y, smoke_pos_z]
    return smoke_pos


def point_to_line_dist(p, a, b):
    """计算点 p 到由 a, b 两点定义的直线的距离"""
    ap = p - a
    ab = b - a
    # 使用向量叉乘计算距离: |ab x ap| / |ab|
    cross_product_norm = np.linalg.norm(np.cross(ab, ap))
    ab_norm = np.linalg.norm(ab)
    if ab_norm == 0:
        return np.linalg.norm(ap)
    return cross_product_norm / ab_norm


# --- 3. 模拟设置和预计算 ---
# 投放时刻
t_drop = T_ASSIGN
# 无人机在投放时刻的位置和速度
p_drop = get_uav_pos(t_drop)
# 烟幕弹继承无人机的速度
## 袁浥原：烟雾弹爆炸之后不再继承无人机的速度
## 显然烟雾弹的爆炸位置和云团的中心位置要分开算。。。


direction_fy_drop = -P_FY1_0 / np.linalg.norm(P_FY1_0)
v_drop = V_FY1 * direction_fy_drop

# 起爆时刻
t_detonate = t_drop + T_INTERVAL
# 起爆点位置
p_detonate = get_bomb_pos(t_detonate, t_drop, p_drop, v_drop)


# --- 4. 遮蔽判断函数 ---
def is_shielded_at_t(t):
    """
    判断在 t 时刻，烟幕是否对 M1 有效遮蔽。
    这是一个完整的状态检查函数。
    """
    # 检查时间是否在有效遮蔽窗口内
    if t < t_detonate or t > t_detonate + SMOKE_DURATION:
        return False

    # a. 计算当前时刻各物体位置
    p_missile = get_missile_pos(t)

    # b. 计算烟幕云团中心位置
    time_after_detonation = t - t_detonate
    p_smoke_center = p_detonate - np.array([0, 0, V_SC_DOWN * time_after_detonation])

    # c. 计算球心到视线的距离

    ## 袁浥原：显然Gemini没有检查真目标的所有点
    distance = point_to_line_dist(p_smoke_center, p_missile, P_TT_BASE)

    ## 袁浥原：  需要添加一个方向检查


    # d. 判断是否遮蔽
    return distance <= R_SC

# --- 5. 主循环与求解 ---
def solve_problem_1():
    """
    通过时间积分计算总遮蔽时长
    """
    total_shield_time = 0.0
    dt = 0.001  # 时间步长，越小越精确
    
    # 确定模拟的时间区间
    start_time = t_detonate
    end_time = t_detonate + SMOKE_DURATION

    # 遍历时间区间内的每一点
    for t in np.arange(start_time, end_time, dt):
        if is_shielded_at_t(t):
            total_shield_time += dt

    return total_shield_time


# --- 6. 运行并输出结果 ---
if __name__ == "__main__":
    # 打印一些关键的预计算结果，便于论文中展示
    print("--- 预计算关键参数 ---")
    print(f"无人机投放时刻 (t_drop): {t_drop:.4f} s")
    print(f"无人机投放位置 (P_drop): {np.round(p_drop, 2)}")
    print(f"烟幕弹起爆时刻 (t_detonate): {t_detonate:.4f} s")
    print(f"烟幕弹起爆位置 (P_detonate): {np.round(p_detonate, 2)}")
    print("-" * 20)

    # 求解问题1
    effective_shield_time = solve_problem_1()

    print("\n--- 问题1 求解结果 ---")
    print(f"烟幕干扰弹对 M1 的有效遮蔽时长为: {effective_shield_time:.4f} s")