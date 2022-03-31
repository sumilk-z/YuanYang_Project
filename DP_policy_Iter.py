import time
import random
from YuanYangEnv import YuanYangEnv

class DP_Policy_Iter(object):
    def __init__(self, yuanyang: YuanYangEnv):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        # 值函数
        # 为何要+1,因为状态是从1号开始的，或许是为了方便
        self.v = [0.0 for i in range(len(self.states) + 10)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        # 初始化策略（随机初始化）
        for state in self.states:
            flag1 = 0
            flag2 = 0
            # 看有没有发生碰撞，若发生碰撞返回1
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 or flag2:
                # 发生碰撞或者已经找到都不需要执行任何动作，因为是终结状态
                continue
            self.pi[state] = self.actions[int(
                random.random() * len(self.actions))]

    # 策略评估
    def policy_evaluate(self):
        # for i in range(100):
        # 迭代次数不指定
        i = 0
        while(True):
            i = i + 1
            delta = 0.0  # 累积差距，用来验证值函数收敛
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(
                    self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(
                    self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                # 这里取出来的是此状态下要进行的行为
                action = self.pi[state]
                next_state, imm_reward, is_terminal = self.yuanyang.transform(
                    state, action)  # 下一个状态，立即回报，以及是否为终结状态
                new_v = imm_reward + self.gamma * \
                    self.v[next_state]  # 更新新的状态值,相当于迭代一次
                delta += abs(self.v[state] - new_v)  # 和原来的值进行比较
                self.v[state] = new_v
            if delta < 1e-6:
                print("经过%d次迭代，状态值已经收敛！" % i)
                break
    # 策略改善：使用贪婪策略

    def poliy_improve(self):
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = self.yuanyang.collide(
                self.yuanyang.state_to_position(state))
            flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue

            # 比较该状态下的每一个动作的q值，选择q值最大的那个动作作为当前状态的策略
            best_action = self.actions[0]
            next_state, imm_reward, is_terminal = self.yuanyang.transform(
                state, best_action)
            best_v = imm_reward + self.gamma * self.v[next_state]
            # 找到使得价值函数最大的那个动作
            for action in self.actions:
                next_state, imm_reward, is_terminal = self.yuanyang.transform(
                    state, action)
                if best_v < imm_reward + self.gamma * self.v[next_state]:
                    best_action = action
                    best_v = imm_reward + self.gamma * self.v[next_state]
            self.pi[state] = best_action  # 贪婪策略更新

    # 策略迭代算法:策略评估和策略改善交叉执行，直到策略不再改变
    def policy_iterate(self):
        i = 0
        while(True):
            i = i + 1
            # 策略评估，直至收敛
            self.policy_evaluate()
            # 保留原来的策略以做比较
            pi_old = self.pi.copy()
            # 策略更新
            self.poliy_improve()
            # 判断策略是否已经收敛
            if self.pi == pi_old:
                print("经过%d次策略迭代，已经收敛到最佳策略！" % i)
                break

if __name__ == '__main__':
    yuanyang = YuanYangEnv()
    policy_value = DP_Policy_Iter(yuanyang)
    policy_value.policy_iterate()

    path = []
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]

    step_sum = 0  # 总共走了多少步

    flag = 1
    s = 0
    while flag:
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_sum += 1
        s_, r, t = yuanyang.transform(s, a)
        if t or step_sum > 200:
            flag = 0
        s = s_
    yuanyang.bird_male_position = yuanyang.state_to_position(s)  # 最后的状态
    path.append(s)
    yuanyang.render()
    print("总共走了%d步！" % step_sum)
    # 保持不动
    while True:
        yuanyang.render()
