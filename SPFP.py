"""
# @Author: JuQi
# @Time  : 2022/8/5 13:51
# @E-mail: 18672750887@163.com
"""
import numpy as np
import joblib
import os
import time
import csv


class Node(object):
    def __init__(self, h, prior_state, now_player):
        self.h = h
        self.is_end = False
        self.now_player = now_player
        self.prior_state = prior_state
        self.utility_matrix = 0
        self.son = {}
        self.action_list = []
        self.action_policy = np.array(0)
        self.is_train = True

    def set_action_list(self, action_list):
        self.action_list = action_list

    def add_son(self, tmp_action, now_player):
        if tmp_action[-1] == '_':
            self.is_train = False
            self.now_player = 'c'
            self.son[tmp_action] = Node(self.h + tmp_action, self.prior_state, now_player)
        else:
            self.action_policy = np.random.random((self.prior_state, len(self.action_list)))
            self.son[tmp_action] = Node(self.h + tmp_action, self.prior_state, now_player)
            self.action_policy = self.action_policy / np.sum(self.action_policy, axis=1).reshape(self.prior_state, 1)

    def set_utility_matrix(self, utility_matrix):
        self.utility_matrix = utility_matrix
        self.is_end = True


class SPFPSolver(object):
    def __init__(self, prior_state, game_name, prior_preference):
        self.prior_state = prior_state
        self.tree_root_node = None
        self.train_num = 1
        self.lr = 1
        self.p0_loss = 0
        self.p1_loss = 0
        self.game_name = game_name
        self.prior_preference = prior_preference

        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/log/', self.game_name, '_', now_time_str])

    def get_now_player(self, h: str) -> str:
        tmp_h = h.split('_')
        if len(tmp_h[-1]) % 2 == 1:
            return 'player1'
        else:
            return 'player0'

    def get_legal_action(self, h: str) -> list:
        pass

    def get_utility_matrix(self, h: str, now_player: str) -> np.ndarray:
        pass

    def tree_dfs(self, now_node: Node):
        print(now_node.h)

        son_list = self.get_legal_action(now_node.h)
        if son_list:
            now_node.set_action_list(son_list)
        else:
            now_node.set_utility_matrix(self.get_utility_matrix(now_node.h, now_node.now_player))
            return

        for son_action in son_list:
            now_node.add_son(son_action, self.get_now_player(now_node.h + son_action))
            self.tree_dfs(now_node.son[son_action])

    def generate_tree(self):
        self.tree_root_node = Node('_', self.prior_state, 'player0')
        self.tree_dfs(self.tree_root_node)

    def flow(self):
        def find_max_action(tmp_result, now_player):
            if now_player == 'player0':
                tmp_BR = (tmp_result == np.max(tmp_result, axis=1)[:, None]).astype(int)
                tmp_BR = tmp_BR / np.sum(tmp_BR, axis=1).reshape(-1, 1)
            elif now_player == 'player1':
                tmp_BR = (tmp_result == np.min(tmp_result, axis=1)[:, None]).astype(int)
                tmp_BR = tmp_BR / np.sum(tmp_BR, axis=1).reshape(-1, 1)
            else:
                tmp_BR = 0
            return tmp_BR

        def flow_dfs(node: Node, p0_policy: np.ndarray, p1_policy: np.ndarray, father_player):
            if not node.is_end:
                if node.now_player == 'player0':
                    tmp_action_result, tmp_utility_matrix = flow_dfs(
                        node.son[node.action_list[0]],
                        node.action_policy[:, 0],
                        p1_policy,
                        node.now_player
                    )
                    for i_son_name in range(1, len(node.action_list)):
                        tmp_p0_result, tmp_p1_matrix = flow_dfs(
                            node.son[node.action_list[i_son_name]],
                            node.action_policy[:, i_son_name],
                            p1_policy,
                            node.now_player
                        )
                        tmp_utility_matrix += tmp_p1_matrix
                        tmp_action_result = np.hstack((tmp_action_result, tmp_p0_result))

                elif node.now_player == 'player1':
                    tmp_action_result, tmp_utility_matrix = flow_dfs(
                        node.son[node.action_list[0]],
                        p0_policy,
                        node.action_policy[:, 0],
                        node.now_player,
                    )
                    for i_son_name in range(1, len(node.action_list)):
                        tmp_p1_result, tmp_p0_matrix = flow_dfs(
                            node.son[node.action_list[i_son_name]],
                            p0_policy,
                            node.action_policy[:, i_son_name],
                            node.now_player
                        )
                        tmp_utility_matrix += tmp_p0_matrix
                        tmp_action_result = np.hstack((tmp_action_result, tmp_p1_result))

                else:
                    tmp_action_result, tmp_utility_matrix = flow_dfs(
                        node.son[node.action_list[0]],
                        p0_policy,
                        p1_policy,
                        node.now_player
                    )
                    for i_son_name in range(1, len(node.action_list)):
                        tmp_p1_result, tmp_p0_matrix = flow_dfs(
                            node.son[node.action_list[i_son_name]],
                            p0_policy,
                            p1_policy,
                            node.now_player
                        )
                        tmp_utility_matrix += tmp_p0_matrix

                # change policy
                if node.now_player != 'c':
                    tmp_max_result = find_max_action(tmp_action_result, node.now_player)
                    this_node_loss = tmp_action_result * (tmp_max_result - node.action_policy)
                    if node.h in self.prior_preference:
                        i_poker = self.prior_preference[node.h][0]
                        i_action = node.action_list.index(self.prior_preference[node.h][1])
                        i_rl = self.prior_preference[node.h][2]
                        if tmp_max_result[i_poker, i_action]:
                            tmp_rl = np.ones(node.prior_state)
                            tmp_rl = tmp_rl * self.lr
                            tmp_rl[i_poker] = tmp_rl[i_poker] * i_rl
                            if tmp_rl[i_poker] > 1:
                                tmp_rl[i_poker] = 1
                            tmp_rl=tmp_rl.reshape(-1,1)
                            node.action_policy = node.action_policy * (1 - tmp_rl) + tmp_rl * tmp_max_result
                        else:
                            node.action_policy = node.action_policy * (1 - self.lr) + self.lr * tmp_max_result
                    else:
                        node.action_policy = node.action_policy * (1 - self.lr) + self.lr * tmp_max_result

                    if node.now_player == 'player0':
                        self.p0_loss = self.p0_loss * p0_policy + np.sum(this_node_loss, axis=1)
                    else:
                        self.p1_loss = self.p1_loss * p1_policy - np.sum(this_node_loss, axis=1)

            else:
                tmp_utility_matrix = node.utility_matrix

            if father_player == 'player0':
                p0_result = np.matmul(tmp_utility_matrix, p1_policy.reshape(self.prior_state, 1))
                p1_matrix = p0_policy.reshape((self.prior_state, 1)) * tmp_utility_matrix
                return p0_result, p1_matrix
            elif father_player == 'player1':
                p1_result = np.matmul(p0_policy, tmp_utility_matrix)
                p0_matrix = p1_policy * tmp_utility_matrix
                return p1_result.reshape((self.prior_state, 1)), p0_matrix
            else:
                return 0, tmp_utility_matrix

        self.p0_loss = 0
        self.p1_loss = 0
        flow_dfs(self.tree_root_node, np.ones(self.prior_state), np.ones(self.prior_state), 'c')
        self.train_num += 1
        self.lr = 1 / self.train_num

    def show_tree(self):
        def show_dfs(node: Node):
            if node.is_end or node.now_player == 'c':
                pass
            else:
                print('node_name  :', node.h)
                print('node_player:', node.now_player)
                print('node_policy:')
                print(node.action_list)
                print(node.action_policy)
                print()
            if node.son:
                for son_name in node.son.keys():
                    show_dfs(node.son[son_name])

        show_dfs(self.tree_root_node)

    def log(self, loop: int, train_info: dict):
        with open(self.result_file_path + '/result.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_info.keys())
            if loop == 0:
                writer.writeheader()
            writer.writerow(train_info)

    def save_model(self, loop):
        joblib.dump(self.tree_root_node, self.result_file_path + '/' + str(loop) + '.pkl')

    def load_model(self, local_dir):
        self.tree_root_node = joblib.load(local_dir)

    def train(self, train_num: int, save_interval: int, log_interval: int):
        os.makedirs(self.result_file_path)
        for episode in range(train_num + 1):
            self.flow()
            if episode % save_interval == 0:
                self.save_model(episode)
            if episode % log_interval == 0:
                print(episode)
                print('loss:', np.sum(self.p0_loss + self.p1_loss) / 3)
                self.log(episode, {'episode': episode, 'loss': np.sum(self.p0_loss + self.p1_loss) / 3})
