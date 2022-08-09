"""
# @Author: JuQi
# @Time  : 2022/8/8 20:49
# @E-mail: 18672750887@163.com
"""

import os
import argparse

import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import tournament

def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

    # Initilize CFR Agent
    agent = CFRAgent(env, os.path.join(args.log_dir, 'cfr_model'))
    agent.load()  # If we have saved model, we first load the model

    # Start training
    for episode in range(args.num_episodes):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with Random agents.
        if episode % args.evaluate_every == 0:
            agent.save() # Save model


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--evaluate_every', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_cfr_result/')
    args = parser.parse_args()
    train(args)
