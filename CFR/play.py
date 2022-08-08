import numpy as np
import game

if __name__ == '__main__':
    # np.random.seed(1)
    # # 训练好的代码进行人机游戏
    # np.random.shuffle(game.poker)
    # # p1是人 p2代表电脑
    # p1 = game.Player(0, game.poker[0], game.AI_state)
    # p2 = game.Player(2, game.poker[1], game.AI_state)
    # game1 = game.Game(p1, p2)
    # game1.flow('', 1)
    # game1.show_result()

    p1 = game.Player(game.poker[0], game.states)
    p2 = game.Player(game.poker[1], game.states)
    game1 = game.Game(p1, p2)
    for i in range(3001):
        np.random.shuffle(game.poker)
        p1.hand = game.poker[0]
        p2.hand = game.poker[1]
        game1.CFR_algorithm('', 1, 1)
        if i % 150 == 0:
            np.save('/Users/juqi/Desktop/居奇综合/all_of_code/SPFP/log_kuhn_CFR/' + str(i)+'.npy', game.CFR_s2)
    print(game.CFR_s2)
