
import random
import numpy as np
from YuanYangEnv import YuanYangEnv
import time


class DP_Value_Iter(object):
    def __init__(self, yuanyang: YuanYangEnv):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for x in range(len(self.states) + 10)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state=state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = self.actions[int(
                random.random() * len(self.actions))]

    def value_iteration(self):
        i = 0
        while(True):
            i = i + 1
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(
                    self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(
                    self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                a1 = self.actions[int(random.random() * 4)]
                s, r, t = self.yuanyang.transform(state, a1)

                v1 = r + self.gamma * self.v[s]
                for action in self.actions:
                    s, r, t = self.yuanyang.transform(state, action)
                    if v1 < r + self.gamma * self.v[s]:
                        a1 = action
                        v1 = r + self.gamma * self.v[s]
                delta += abs(v1 - self.v[state])
                self.pi[state] = a1
                self.v[state] = v1
            if delta < 1e-6:
                print("经过%d次策略评估和迭代，策略已经收敛，得到最佳策略！" % i)
                break


if __name__ == '__main__':
    yuanyang = YuanYangEnv()
    policy_value = DP_Value_Iter(yuanyang)
    policy_value.value_iteration()

    s = 0
    path = []
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    flag = 1
    step_num = 0

    while flag:
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t or step_num > 20:
            flag = 0
        s = s_
    print("总共走了%d步！"%step_num)
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
