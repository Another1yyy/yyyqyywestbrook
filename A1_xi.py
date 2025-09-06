from os import name

import numpy as np
import math as math
import config
from A2LONG import effective_span
### 初始化变量
import item
time = 0.00
g = 9.81              # 重力加速度 (m/s^2)
v_m = 300.0          # 导弹速度 (m/s)
R_cloud = 10.0       # 云团有效半径 (m)
sink_v = 3.00         # 云团下沉速度 (m/s)
effective_span = 20.0  # 起爆后有效 20 s

# 真目标与导弹初始
T = np.array([0.0, 200.0, 10.0])          # 真目标
M0 = np.array([20000.0, 0.0, 2000.0])    # 导弹 M1 初始位置，指向原点
fake_target = np.array([0.0, 0.0, 0.0])
# FY1 无人机：初始、航向、速度、投放与起爆设置
F0 = np.array([17800.0, 0.0, 1800.0])    # FY1 初始
vec = np.array([0,0,1800]) - F0


h1 = np.array([-1,0,0])
  #h是飞机的航向
v_u = 120.0                                  # FY1 速度
t_drop = 1.50                             # 受令后投放时刻 (s)
t_explosion = 3.60
t1_start = t_drop + t_explosion ##云开始下沉
dt = 0.001
###
def main():
    # a. 更新所有活动对象的位置
  time = 0.0
  shield_time = 0.0
  TARGET_BASE = np.array([0.0, 200.0, 0.0])
  TARGET_R = 7.0
  TARGET_H = 10.0
  R_cloud = 10.0

   # 生成目标采样点
  CYL_PTS = sample_cylinder_points(TARGET_BASE, TARGET_R, TARGET_H)
  flight = item.plane(1, F0, v_u, h1, t_drop, t_explosion)
  cloud1 = None
  bomb1 = None
  bomb_dropped = False
  missile1 = item.missile(time, M0, v_m)
  simulation_duration = effective_span + t_drop + t_explosion
  while time < simulation_duration:
    flight.move(time)
    missile1.move(time)

    if time >= t_drop and not flight.released and not bomb_dropped:
        bomb1 = flight.release_bomb(time)
        print(f"Bomb object ID: {id(bomb1)}")
        flight.released = True
        bomb_dropped = True

        print(f"[{time:.2f}s] 飞机投放了炸弹。")
        print(f"投弹位置L:{flight.pos},{bomb1.pos}")
    if bomb1 is None and time >= t1_start:
        print("发现错误炸弹为空")
    if  bomb1 and time >= t1_start and not cloud1:
        print("进入云团爆炸逻辑")
        cloud1 = bomb1.explose(time)
        print(f"[{time:.2f}s] 炸弹起爆，形成云团。")
        print(f"云团位置：{bomb1.pos},{cloud1.pos} ")
    if bomb1:
        bomb1.move(time)
    if cloud1:
        cloud1.move(time)

    # b. 处理一次性事件 (投放 和 起爆)
    # 投放事件
    # if time >= 5.10:
    #     print(f"导弹位置：{missile1.pos}")

        # c. 检查连续状态 (是否遮蔽)
    if cloud1 and t1_start <= time <= t1_start + effective_span:
        # 【核心修正】判断条件应该是导弹当前位置 missile1.pos 和 真目标 T
        # print("时间条件满足")
        #这里是可以进来的
        # if is_shield_physical(missile1.pos, T, cloud1.pos, R_cloud):
        #     shield_time += dt
        #     print("嘿！程序运行到了sheildtime累加环节")
        # if is_shield_physical(cloud1.pos, T,missile1.pos, R_cloud):
        #         shield_time += dt
        #         print("嘿！程序运行到了sheildtime累加环节")
        # if check_shielding_with_volume_target(missile1.pos,cloud1.pos,CYL_PTS,R_cloud)["is_shielded"]:
        #     shield_time += dt
        #     print("进入新情况下的shield")
        if is_shield(cloud1.pos,missile1.pos,T,R_cloud):
            print("jin")
            shield_time += dt

        # d. 时间前进
    time += dt
  print(f"main循环内的总时间[{time:.2f}s]")
  print(f"调试t1start：{t1_start:.2f}s")
  return shield_time
def is_shield(cloud_center,missile_pos,real_pos,cover_range):
    missile_to_target = real_pos - missile_pos
    missile_to_cloud = cloud_center - missile_pos
    chengji  = float(np.dot(missile_to_cloud , missile_to_target))
    square = float(np.dot(missile_to_target,missile_to_target ))
    rate = chengji / square
    rate_final = min(1,max(0,rate))
    Q = missile_pos + rate_final * missile_to_target
    ## Q点计算不正确
    distance = np.linalg.norm(cloud_center - Q)
    print(distance)
    if distance <= cover_range:
        return True
    else :
        return False











### 变量初始化直接在config文件里

###思路：
###part1：核心判断是烟雾团是否和目标导弹的连线有交点
###导弹的点进行遍历来判断直线，烟雾中心和直线的距离来判断是否遮蔽
def is_shield_physical(A,B,center,max_cover):
    if center is None:
        print("云团消失辣")
        return False
    distance = point_line_dis(center,A,B)
    ### checked
    if distance < max_cover:
        return True
    else:
        return False

## 这里其实不太聪明，就是drop的时间加上explosion之后才能遮蔽
###问题在于，应该对于单架飞机做一个持续判断的程序
def is_shield_conditional(A,B,center,max_cover,time,t_drop,t_explosion):
    ###先检查时间条件
    if time < t_drop + t_explosion or time > t_drop + t_explosion + effective_span:
        return False
    ###最后检查空间条件
    if is_shield_physical(A,B,center,max_cover):
        return True


def mainloop():
    time = 0.0
    loop_end = False
    while True:

        time += 0.001
##number 为cloud的序号，写的时候产生的几个云团要分开
def cloud_center(init_pos,t,number):
    pos = init_pos
    pos[2] = init_pos[2] + 3.0 * t

    return pos




def point_line_dis(point,A,B):
     BA = B - A
     ## BA是从a到b的向量
     nearest_point = A
     l2 = float(np.dot(BA,BA))
     ## l2是bA直线的模长
     if l2 == 0.0:
         return float(np.linalg.norm(point-A))
     else:
         s = float(np.dot(point-A,BA) / l2)
         s_rate = max(0.0,min(1.0,s))
         Q = A + s_rate * BA
         value = float(np.linalg.norm(point-Q))
         return value


# 2. 圆柱体采样函数
# =========================
def sample_cylinder_points(base, radius, height,
                           n_theta_side=32, n_z_side=5,
                           n_theta_disk=24, n_r_disk=3):
    """生成圆柱体上的离散采样点"""
    pts = []
    thetas_side = np.linspace(0, 2 * np.pi, n_theta_side, endpoint=False)
    zs = np.linspace(0, height, n_z_side + 1)
    # 侧面采样
    for th in thetas_side:
        c, s = np.cos(th), np.sin(th)
        for z in zs:
            pts.append([base[0] + radius * c, base[1] + radius * s, base[2] + z])
    # 顶面和底面采样
    for z_level in [0.0, height]:
        thetas_disk = np.linspace(0, 2 * np.pi, n_theta_disk, endpoint=False)
        rs = np.linspace(radius / (n_r_disk + 1), radius, n_r_disk)
        for r in rs:
            for th in thetas_disk:
                c, s = np.cos(th), np.sin(th)
                pts.append([base[0] + r * c, base[1] + r * s, base[2] + z_level])
    # 增加中心轴线上的点
    pts.append(base)
    pts.append(base + np.array([0, 0, height]))
    return np.asarray(pts, dtype=float)


# =========================
# 3. 核心遮蔽判断函数 (调用批量计算)
# =========================
def check_shielding_with_volume_target(missile_pos, smoke_center, target_samples, smoke_radius):
    """
    使用批量计算来判断是否发生遮蔽。
    """
    # 准备批量计算的输入
    num_samples = len(target_samples)
    missile_pos_batch = np.repeat(missile_pos[None, :], num_samples, axis=0)

    # 调用批量距离计算函数
    distances, projection_params = dist_point_to_segments_batch(
        smoke_center, missile_pos_batch, target_samples
    )

    # 找到最小距离
    min_distance = np.min(distances)

    # 判断是否遮蔽
    is_shielded = min_distance <= smoke_radius

    # 返回详细信息，便于分析
    return {
        "is_shielded": is_shielded,
        "min_distance": min_distance,
        "closest_point_index": np.argmin(distances),
        "distances": distances,
        "projection_params": projection_params
    }

def dist_point_to_segments_batch(P, Ms, Xs):
    """
    单点 P 到一批线段 [M_i, X_i] 的距离（向量化）
    Ms, Xs: (N,3)
    返回: distances(N,), proj_s(N,)  (投影参数 ∈[0,1])
    """
    BA = Xs - Ms                # (N,3)
    BA2 = np.einsum('ij,ij->i', BA, BA)  # (N,)
    # 处理零长线段的稳健性
    zero = BA2 < 1e-12
    BA2[zero] = 1.0  # 避免除0，稍后用特别分支修正

    PA = P[None, :] - Ms        # (N,3)
    s = np.einsum('ij,ij->i', PA, BA) / BA2
    s = np.clip(s, 0.0, 1.0)
    Q = Ms + s[:, None] * BA    # (N,3)
    d = np.linalg.norm(P[None, :] - Q, axis=1)

    # 零长线段：距离退化为 |P - M|
    if np.any(zero):
        d[zero] = np.linalg.norm(P[None, :] - Ms[zero], axis=1)
        s[zero] = 0.0
    return d, s


if __name__ == "__main__":
    print("程序开始运行")
    shield_time = main()
    print(f"遮蔽时间：{shield_time}")

