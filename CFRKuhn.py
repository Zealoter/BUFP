"""
# @Author: JuQi
# @Time  : 2022/8/8 19:32
# @E-mail: 18672750887@163.com
"""
"""
# @Author: JuQi
# @Time  : 2022/7/26 14:55
# @E-mail: 18672750887@163.com
"""
import numpy as np
from SPFP import SPFPSolver
import csv


class KuhnPokerSolver(SPFPSolver):
    def __int__(self, prior_state, name, prior_preference):
        super(KuhnPokerSolver, self).__init__(prior_state, name, prior_preference)

    def get_legal_action(self, h: str) -> list:
        if h[-1] == '_':
            return ['R', 'C']
        elif h[-1] == 'R':
            return ['F', 'C']
        elif h[-2:] == '_C':
            return ['C', 'R']
        elif h[-1] == 'F':
            return []

        else:
            return []

    def get_utility_matrix(self, h: str, now_player: str) -> np.ndarray:
        tmp_h = h.split('_')
        money = 1
        money = money + tmp_h[1].count('R')
        need_eye = 1 - np.eye(self.prior_state)
        if tmp_h[-1][-2:] == 'RF':
            money = money - 1
            if now_player == 'player0':
                return money * need_eye
            elif now_player == 'player1':
                return (-money) * need_eye

        elif tmp_h[-1][-2:] == 'CC' or tmp_h[-1][-2:] == 'RC':
            result = np.array([
                [0, -1, -1],
                [1, 0, -1],
                [1, 1, 0]
            ])
            return result * money


def node_trans(tmp_node):
    tmp_tans = {
        '_'  : '',
        '_R' : 'b',
        '_C' : 'p',
        '_CR': 'pb',
    }
    for i in range(3):
        tmp_info_name = str(i + 1) + tmp_tans[tmp_node.h]
        if tmp_node.h == '_':
            tmp_node.action_policy[i, 1] = tmp_policy[tmp_info_name][0]
        else:
            tmp_node.action_policy[i, 1] = tmp_policy[tmp_info_name][1]
        tmp_node.action_policy[i, 0] = 1 - tmp_node.action_policy[i, 1]


def dfs(node):
    if node.is_end:
        return
    node_trans(node)
    for i_s in node.action_list:
        dfs(node.son[i_s])


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    loop = 0
    while loop <= 3000:
        tmp = KuhnPokerSolver(3, 'Kuhn_Poker_CFR', {})
        tmp.generate_tree()
        tmp_policy = np.load('/Users/juqi/Desktop/居奇综合/all_of_code/SPFP/log_kuhn_CFR/' + str(loop) + '.npy',
                             allow_pickle=True).item()
        dfs(tmp.tree_root_node)
        tmp.show_tree()
        tmp.flow(is_train=False)
        train_info = {'episode'     : loop, 'loss': np.sum(tmp.p0_loss + tmp.p1_loss) / 3,
                      'subgame_loss': np.sum(tmp.p0_loss_subgame + tmp.p1_loss_subgame) / 3}
        with open('/Users/juqi/Desktop/居奇综合/all_of_code/SPFP/log_kuhn_CFR/result.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_info.keys())
            if loop == 0:
                writer.writeheader()
            writer.writerow(train_info)
        loop += 150
