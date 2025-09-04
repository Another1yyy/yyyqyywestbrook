import numpy as np

def calculate_explosion_pos(V_FY,T_INTERVAL,pos_start):
    pos_xyz = V_FY[:3] * T_INTERVAL + pos_start[:3]
    return pos_xyz
def calculate_smoke_pos(explosion_pos,t_drop):
    smoke_pos_z = explosion_pos[2] + 3.00 * t_drop
    smoke_pos_x = explosion_pos[0]
    smoke_pos_y = explosion_pos[1]
    smoke_pos = [smoke_pos_x, smoke_pos_y, smoke_pos_z]
    return smoke_pos