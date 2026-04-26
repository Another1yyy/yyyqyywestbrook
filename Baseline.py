
import numpy as np

from Q1_3 import missile_pos, drone_pos
C = np.array([0.0, 200.0, 0.0])
def dist_to_point(A,B,C): #A是导弹位置，B是无人机位置，C是真目标位置
    AC = C-A
    BC = C-B
    AC_length = np.einsum('ij,ij->i',AC,AC)
    BC_length = np.einsum('ij,ij->i',BC,BC)
    return AC_length,BC_length

def cal_baseline(t, v_u, v_m):
    MP = missile_pos(t)
    DP = drone_pos(t)
    missile_dist, drone_dist = dist_to_point(MP,DP,C)
    if missile_dist <= d
