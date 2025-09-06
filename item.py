g = 9.81
import numpy as np


class missile():
    def __init__(self, number, start_pos, speed_magnitude):
        """
        初始化一枚导弹。

        :param number: 导弹编号 (e.g., "M1")
        :param start_pos: 导弹的初始三维位置 (numpy array)
        :param speed_magnitude: 导弹的速度大小 (标量, e.g., 300.0 m/s)
        :param target_pos: 导弹的攻击目标位置 (numpy array, e.g., 假目标原点)
        """
        self.number = number
        self.start_pos = np.array(start_pos, dtype=float)
        self.pos = np.copy(self.start_pos)  # 当前位置
        self.speed_magnitude = float(speed_magnitude)
        self.target_pos = np.zeros(3)

        # 计算方向向量：从初始位置指向目标位置
        direction_vector = self.target_pos - self.start_pos

        # 归一化，得到单位方向向量
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            # 防止除以零，如果起点和终点重合，则方向为(0,0,0)
            self.speed_dir = np.zeros(3)
        else:
            self.speed_dir = direction_vector / norm

        # 计算速度矢量
        self.speed = self.speed_magnitude * self.speed_dir

    def move(self, time):
        """
        根据从0开始的总时间 t，更新导弹的位置。
        这是一个绝对位置更新，而不是增量更新。

        :param time: 从 t=0 开始经过的总时间
        """
        # 导弹做匀速直线运动
        self.pos = self.start_pos + self.speed * time


class plane():
    def __init__(self, number,start_pos,speed_magnitude,speed_direction,release_time,explosion_time):
        self.number = number
        self.start_pos = start_pos
        self.pos = start_pos
        self.speed_magnitude = speed_magnitude
        self.speed_dir = speed_direction
        self.speed = speed_magnitude * speed_direction
        self.bomb_time = release_time + explosion_time
        self.explosion_time = explosion_time
        self.released = False

    def move(self,time):
        self.pos = self.start_pos + self.speed * time
    def release_bomb(self,time):

            self.pos = self.start_pos + self.speed * time
            new_bomb = bomb(self.number,time,self.pos,self.speed,self.explosion_time)

            self.released = True
            return new_bomb


class bomb():
    def __init__(self, init_time,number,start_pos,speed,explosion_time):
        self.number = number
        self.init_time = init_time
        self.start_pos = start_pos
        self.speed = speed
        self.explosion_time = explosion_time
        self.moving = True
        self.pos = self.start_pos
    def move(self,time):
        if time > self.init_time + self.explosion_time:
            self.moving = False
        if self.moving == True:
            moving_time = time - self.init_time

            pos = self.start_pos + self.speed * moving_time
            pos[2] = self.start_pos[2] - 0.5 * g * moving_time ** 2 + self.speed[2] * moving_time


            self.pos = pos
    def explose(self,time):
        new_cloud  = cloud(time,self.number,self.pos)
        return new_cloud

class cloud():
    def __init__(self,init_time,number,start_pos):
        self.init_time = init_time
        self.number = number
        self.start_pos = start_pos
        self.pos = self.start_pos
    def move(self,time):
        elapsed_time = time - self.init_time
        self.pos[2] = self.pos[2] - elapsed_time * 3.0

        ###3.0为云团下降速度





