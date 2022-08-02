"""
# @Author: JuQi
# @Time  : 2022/7/25 11:43
# @E-mail: 18672750887@163.com
"""

from pettingzoo.classic import leduc_holdem_v3
import numpy as np
from FPSBLedec import FPSBSolver
from FPSBLedec import Node


def print_info(agent, observation, reward, done, info):
    print(agent)
    print('手牌：', end='')
    if observation['observation'][0] == 1:
        print('J')
    elif observation['observation'][1] == 1:
        print('Q')
    else:
        print('K')
    print('公共牌：', end='')
    if observation['observation'][3] == 1:
        print('J')
    elif observation['observation'][4] == 1:
        print('Q')
    elif observation['observation'][5] == 1:
        print('K')
    else:
        print('暂无')
    if agent == 'player_1':
        print('player_1', np.argmax(observation['observation'][6:20]))
        print('player_0', np.argmax(observation['observation'][21:35]))
    else:
        print('player_1', np.argmax(observation['observation'][21:35]))
        print('player_0', np.argmax(observation['observation'][6:20]))
    print(done)
    print(reward)
    print()
    # print(info)


def get_my_card(tmp_ob):
    if tmp_ob['observation'][0]:
        return 0
    elif tmp_ob['observation'][1]:
        return 1
    else:
        return 2


def get_table_card(tmp_ob):
    if tmp_ob['observation'][3]:
        return '_J_'
    elif tmp_ob['observation'][4]:
        return '_Q_'
    elif tmp_ob['observation'][5]:
        return '_K_'
    else:
        return 0


if __name__ == '__main__':
    agent = FPSBSolver(3)
    agent.load_model('/Users/juqi/Desktop/居奇综合/all_of_code/texas/log/Leduc_Poker_2022_07_27_20_09_31/10000.pkl')
    env = leduc_holdem_v3.env()
    action_list = [1, 1, 3, 1, 3, 1, 3, 3]

    action_code = {'C': 3, 'R': 1, 'F': 2}
    action_decode = ['C', 'R', 'F', 'C']
    ans = 0
    for _ in range(1000):
        is_2 = False
        work_tree = agent.tree_root_node
        now_action = 0
        env.reset()
        for agent_name in env.agent_iter():

            observation, reward, done, info = env.last()
            # print_info(agent_name, observation, reward, done, info)

            if done:
                ans += env.rewards['player_0']
                # print(env.rewards)
                break
            if not is_2 and get_table_card(observation):
                is_2 = True
                work_tree = work_tree.son[get_table_card(observation)]

            if agent_name == 'player_0':
                my_card = get_my_card(observation)
                prob = work_tree.action_policy[my_card, :]
                action1 = np.random.choice(work_tree.action_list, p=prob)
                action = action_code[action1]
                work_tree = work_tree.son[action1]
            else:
                action = action_list[now_action]
                work_tree = work_tree.son[action_decode[action]]

            now_action += 1

            if observation['action_mask'][action] == 1:
                env.step(action)
            elif action == 3:
                env.step(0)
    print(ans)
