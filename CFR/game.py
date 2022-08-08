import numpy as np

# 牌
poker = [1, 2, 3]
# 存储状态
in_game_state = ['1', '1b', '1p', '1pb', '2', '2b', '2p', '2pb', '3', '3b', '3p', '3pb']

states = {'1' : [0.5, 0.5], '1b': [0.5, 0.5], '1p': [0.5, 0.5], '1pb': [0.5, 0.5], '2': [0.5, 0.5], '2b': [0.5, 0.5],
          '2p': [0.5, 0.5], '2pb': [0.5, 0.5], '3': [0.5, 0.5], '3b': [0.5, 0.5], '3p': [0.5, 0.5], '3pb': [0.5, 0.5]}

end = ['pp', 'pbp', 'pbb', 'bp', 'bb']

CFR_r = {'1' : [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
         '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}

pass_bet = {'p': 0, 'b': 1}

action = ['p', 'b']

# CFR中间状态
CFR_s = {'1' : [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
         '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}
CFR_s2 = {'1' : [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
          '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}


class Player(object):
    def __init__(self, hand, decision):
        self.hand = hand  # 手牌
        self.information_set = str(hand)  # 所处信息集
        self.decision = decision  # 策略集上的概率分布

    # 选择出牌
    def choice(self, h):
        return self.AI_choice(h)

    def AI_choice(self, h):
        a = np.random.choice(['p', 'b'], p=self.decision[str(self.hand) + h])

        self.information_set += a
        return a


class Game(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.end = 0
        self.result = []

    # 游戏流程
    def flow(self, h, display):
        if h in end:
            self.end = 1
            self.result = self.judge(h)
            return
        if len(h) % 2 == 0:
            a = self.p1.choice(h, display=display)
        else:
            a = self.p2.choice(h, display=display)
        self.flow(h + a, display)

    # 判断胜负
    def judge(self, h):
        if h == 'pp':
            if self.p1.hand > self.p2.hand:
                return [1, -1]
            else:
                return [-1, 1]

        if h == 'pbp':
            return [-1, 1]

        if h == 'pbb' or h == 'bb':
            if self.p1.hand > self.p2.hand:
                return [2, -2]
            else:
                return [-2, 2]

        if h == 'bp':
            return [1, -1]

    # 显示结果
    def show_result(self):
        print('result:', self.result)
        print('p1 hand:', self.p1.hand)
        print('p2 hand:', self.p2.hand)

    # CFR流程
    def CFR_algorithm(self, h, pai1, pai2):
        if h in end:
            tmp = self.judge(h)
            return tmp

        if len(h) % 2 == 0:
            tmp_h = str(self.p1.hand) + h
        else:
            tmp_h = str(self.p2.hand) + h
        va = [0, 0]
        for a in action:
            # 玩家1
            if len(h) % 2 == 0:
                tmp_va = self.CFR_algorithm(h + a, pai1 * states[tmp_h][pass_bet[a]], pai2)
                va[pass_bet[a]] = tmp_va[0]
            else:
                tmp_va = self.CFR_algorithm(h + a, pai1, pai2 * states[tmp_h][pass_bet[a]])
                va[pass_bet[a]] = tmp_va[1]
        # 平均虚拟效用
        ave_va = states[tmp_h][0] * va[0] + states[tmp_h][1] * va[1]

        if len(h) % 2 == 0:
            oppo_pai = pai2
            self_pai = pai1
        else:
            oppo_pai = pai1
            self_pai = pai2

        CFR_r[tmp_h][0] = CFR_r[tmp_h][0] + oppo_pai * (va[0] - ave_va)
        CFR_r[tmp_h][1] = CFR_r[tmp_h][1] + oppo_pai * (va[1] - ave_va)
        CFR_s[tmp_h][0] = CFR_s[tmp_h][0] + self_pai * states[tmp_h][0]
        CFR_s[tmp_h][1] = CFR_s[tmp_h][1] + self_pai * states[tmp_h][1]
        CFR_s2[tmp_h][0] = CFR_s[tmp_h][0] / (CFR_s[tmp_h][0] + CFR_s[tmp_h][1])
        CFR_s2[tmp_h][1] = CFR_s[tmp_h][1] / (CFR_s[tmp_h][0] + CFR_s[tmp_h][1])

        if len(h) % 2 == 0:
            self.change_states(h, 1)
            return [ave_va, -ave_va]
        else:
            self.change_states(h, 2)
            return [-ave_va, ave_va]

    def change_states(self, h, p):
        if p == 1:
            tmp_h = str(self.p1.hand) + h
        else:
            tmp_h = str(self.p2.hand) + h

        p = max([CFR_r[tmp_h][0], 0])
        b = max([CFR_r[tmp_h][1], 0])
        if p == 0 and b == 0:
            states[tmp_h] = [0.5, 0.5]
        else:
            states[tmp_h] = [p / (p + b), b / (p + b)]
