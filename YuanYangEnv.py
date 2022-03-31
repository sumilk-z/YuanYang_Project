
import numpy as np
import pygame
import load
import random

class YuanYangEnv(object):
    def __init__(self):
        # 状态空间
        self.states = [x for x in range(100)]  # 100个状态空间，对应100个移动位置
        # 动作空间
        self.actions = ['e', 's', 'w', 'n']
        # 每个状态处所对应的值函数，总共100个状态
        self.value = np.zeros((10, 10))
        self.gamma = 0.8  # 折扣值
        # 渲染窗口
        self.viewer = None
        # create an object to help track time
        self.FPSLOCK = pygame.time.Clock()
        # 屏幕大小
        self.screen_size = (1200, 800)
        # 雄鸟初始位置
        self.bird_position = (0, 0)
        # 每次步长
        self.limit_distance_x = 120
        self.limit_distance_y = 80
        # 每一个方块障碍物的大小
        self.obstacle_size = [120, 80]
        # 第一堵墙
        self.obstacle1_x = []
        self.obstacle1_y = []
        # 第二堵墙
        self.obstacle2_x = []
        self.obstacle2_y = []

        # 新增横向的墙
        self.obstacle3_x = [600,960, 1080]
        self.obstacle3_y = [400,320, 320]

        # 每一堵墙有8个方块，但是不连续的。
        for i in range(8):
            # obstical 1
            self.obstacle1_x.append(360)
            if i <= 3:
                # 以左上角为方块的坐标
                self.obstacle1_y.append(80 * i)
            else:
                self.obstacle1_y.append(80 * (i + 2))
            # obstical 2
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(80 * i)
            else:
                self.obstacle2_y.append(80 * (i + 2))

        self.bird_male_init_position = (0.0, 0.0)
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [1080, 0]

        self.path = []  # 路径

    # 雄鸟碰撞检测函数
    def collide(self, state_position):
        flag = 1
        flag1 = 1
        flag2 = 1
        flag3 = 1
        # 判断第一个 障碍物
        dx = []
        dy = []
        for i in range(len(self.obstacle1_y)):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle2_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0

        # 判断第二个障碍
        second_dx = []
        second_dy = []
        for i in range(len(self.obstacle2_x)):
            dx2 = abs(self.obstacle2_x[i] - state_position[0])
            second_dx.append(dx2)
            dy2 = abs(self.obstacle2_y[i] - state_position[1])
            second_dy.append(dy2)
        mindx = min(second_dx)
        mindy = min(second_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0
        # 判断第三个障碍物
        third_dx = []
        third_dy = []
        for i in range(len(self.obstacle3_y)):
            dx3 = abs(self.obstacle3_x[i] - state_position[0])
            third_dx.append(dx3)
            dy3 = abs(self.obstacle3_y[i] - state_position[1])
            third_dy.append((dy3))
        mindx = min(third_dx)
        mindy = min(third_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag3 = 0

        if flag1 == 0 and flag2 == 0 and flag3 == 0:
            flag = 0
        # 出界判断
        if state_position[0] > 1080 or\
                state_position[0] < 0 or\
                state_position[1] < 0 or state_position[1] > 810:
            flag = 1
        return flag

    # 雄鸟找到雌鸟
    def find(self, state_position):
        flag = 0
        # if abs(
        #         state_position[0] -
        #         self.bird_female_init_position[0]) < self.limit_distance_x and abs(
        #         state_position[1] -
        #         self.bird_female_init_position[1]) < self.limit_distance_y:
        #     flag = 1
        if(state_position[0] == self.bird_female_init_position[0]
                and state_position[1] == self.bird_female_init_position[1]):
            flag = 1
        return flag

    # 重置子函数
    def reset(self):
        flag1 = 1  # 1表示没找到
        flag2 = 1
        # 随机初始化一个状态，但是这个状态不能是找到雌鸟，也不能是撞到墙
        # 必须flag1=0 且 flag2=0才能结束while循环
        while flag1 or flag2:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collide(state_position)
            flag2 = self.find(state_position)
        return state

    # 状态转为像素坐标
    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 120 * j
        position[1] = 80 * i
        return position

    # 像素坐标转为状态
    def position_to_state(self, position):
        i = position[0] / 120
        j = position[1] / 80
        return int(i + 10 * j)

    # 状态转移函数，状态state下采取动作action到达下一个状态，立即回报，是否终结
    def transform(self, state, action):
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        flag_collide = 0
        flag_find = 0
        # 判断碰撞
        flag_collide = self.collide(current_position)
        # 判断终点
        flag_find = self.find(current_position)

        if flag_find == 1:
            # 找到，奖励为1
            return state, 1, True
        if flag_collide == 1:
            # 没找到，奖励为-1
            return state, -1, True

        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]

        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 80

        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]

        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 80
        flag_collide = self.collide(next_position)
        flag_find = self.find(next_position)
        if flag_collide == 1:
            return self.position_to_state(current_position), -1, True

        if flag_find == 1:
            return self.position_to_state(next_position), 1, True

        return self.position_to_state(next_position), 0, False

    # 以上的功能实现的是Markov决策过程中的所有的关键因素
    # 游戏结束控制函数
    # TODO 这里有一个问题，如果按照源码当中的QUIT会报错
    def gameover(self):
        for event in pygame.event.get():
            # pygame.event.get() return a queue
            if event.type == pygame.QUIT:
                exit()

    # 游戏渲染
    def render(self):
        if self.viewer is None:
            pygame.init()
        self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
        pygame.display.set_caption('DeepLearning Practice')
        # 加载图片
        self.bird_male = load.load_bird_male()
        self.bird_female = load.load_bird_female()
        self.background = load.load_background()
        self.obstacle = load.load_obstacle()
        self.viewer.blit(self.background, (0, 0))
        self.font = pygame.font.SysFont('Times New Roman', 15)  # times
        self.viewer.blit(self.background, (0, 0))
        # 画直线
        for i in range(11):
            # 横竖各11条直线
            # closed=True表示收尾连接 元组为points，1表示线条粗细
            pygame.draw.lines(self.viewer, (255, 255, 255),
                              True, ((120 * i, 0), (120 * i, 800)), 1)
            pygame.draw.lines(self.viewer, (255, 255, 255),
                              True, ((0, 80 * i), (1200, 80 * i)), 1)
        self.viewer.blit(self.bird_female, self.bird_female_init_position)
        # 画墙
        for i in range(8):
            self.viewer.blit(
                self.obstacle,
                (self.obstacle1_x[i],
                 self.obstacle1_y[i]))
            self.viewer.blit(
                self.obstacle,
                (self.obstacle2_x[i],
                 self.obstacle2_y[i]))
        #横着的墙
        for i in range(len(self.obstacle3_x)):
            self.viewer.blit(self.obstacle,(self.obstacle3_x[i],self.obstacle3_y[i]))
        self.viewer.blit(self.bird_male, self.bird_male_init_position)
        # 画值函数
        for i in range(10):
            for j in range(10):
                surface = self.font.render(
                    str(round(float(self.value[i, j]), 3)), True, (0, 0, 0))
                self.viewer.blit(
                    surface, (120 * i + 5, 80 * j + 60))  # 画到方框的下边缘

        # 画路径点
        for i in range(len(self.path)):
            rec_position = self.state_to_position(self.path[i])
            pygame.draw.rect(
                self.viewer, [
                    255, 0, 0], [
                    rec_position[0], rec_position[1], 120, 80], 3)  # width=0 表示fill the rect
            surface = self.font.render(
                str(i + 1), True, (255, 0, 0))  # 画状态的编号，抗锯齿
            self.viewer.blit(
                surface, (rec_position[0] + 5, rec_position[1] + 5))
        pygame.display.update()
        self.FPSLOCK.tick(30) #每秒30帧


if __name__ == '__main__':
    yy = YuanYangEnv()
    yy.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # q结束游戏
                if event.key == pygame.K_q:
                    exit()
            #点x也能结束游戏
            if event.type == pygame.QUIT:
                exit()
