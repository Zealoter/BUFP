"""
# @Author: JuQi
# @Time  : 2022/7/26 14:55
# @E-mail: 18672750887@163.com
"""
import numpy as np
from XFP import XFPSolver


class LeducPokerSolver(XFPSolver):
    def __int__(self, prior_state, game_name, prior_preference):
        super(LeducPokerSolver, self).__init__(prior_state, game_name, prior_preference)

    def get_legal_action(self, h: str) -> list:
        if h[-1] == '_':
            return ['R', 'C']
        elif h[-2:] == 'RR':
            return ['F', 'C']
        elif h[-1] == 'R':
            return ['F', 'C', 'R']
        elif h[-2:] == '_C':
            return ['C', 'R']
        elif h[-1] == 'F':
            return []
        elif h[-1] == 'C':
            tmp_h = h.split('_')
            if len(tmp_h) == 2:
                return ['_J_', '_Q_', '_K_']
            else:
                return []
        else:
            return []

    def get_utility_matrix(self, h: str, now_player: str) -> np.ndarray:
        tmp_h = h.split('_')
        money = 1
        money = money + tmp_h[1].count('R')
        if len(tmp_h) == 4:
            money = money + 2 * tmp_h[3].count('R')
        need_eye = 1 - 0.5 * np.eye(self.prior_state)

        if tmp_h[-1][-2:] == 'RF':
            if len(tmp_h) == 4:
                money -= 2
            else:
                money -= 1
            if len(tmp_h) == 2:
                if now_player == 'player0':
                    return money * need_eye
                elif now_player == 'player1':
                    return (-money) * need_eye
            else:
                if tmp_h[2] == 'J':
                    result = np.array([
                        [0, 0.5, 0.5],
                        [0.5, 0, 1],
                        [0.5, 1, 0]
                    ])
                elif tmp_h[2] == 'Q':
                    result = np.array([
                        [0, 0.5, 1],
                        [0.5, 0, 0.5],
                        [1, 0.5, 0]
                    ])
                else:
                    result = np.array([
                        [0, 1, 0.5],
                        [1, 0, 0.5],
                        [0.5, 0.5, 0]
                    ])

                if now_player == 'player0':
                    return money * result
                elif now_player == 'player1':
                    return (-money) * result

        elif tmp_h[-1][-2:] == 'CC' or tmp_h[-1][-2:] == 'RC':
            if tmp_h[2] == 'J':
                result = np.array([
                    [0, 0.5, 0.5],
                    [-0.5, 0, -1],
                    [-0.5, 1, 0]
                ])
            elif tmp_h[2] == 'Q':
                result = np.array([
                    [0, -0.5, -1],
                    [0.5, 0, 0.5],
                    [1, -0.5, 0]
                ])
            else:
                result = np.array([
                    [0, -1, -0.5],
                    [1, 0, -0.5],
                    [0.5, 0.5, 0]
                ])
            return result * money


if __name__ == '__main__':
    test_list = [
        ['Leduc_Poker', {}]
    ]
    for i_list in test_list:
        np.set_printoptions(precision=6, suppress=True)
        tmp = LeducPokerSolver(3, i_list[0], i_list[1])
        tmp.generate_tree()
        tmp.train(10000, 200, 200)
        tmp.show_tree()
        print(tmp.tree_root_node.action_list)
        print(tmp.tree_root_node.action_policy)

        # for _ in range(10000):
    #     tmp.flow()
    # tmp.show_tree()
