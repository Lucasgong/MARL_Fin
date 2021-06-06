#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python main.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000

Example cartpole command (~8k ts to solve):
    python main.py \
        --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
        --replay_size 5000 --starts_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
        --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
"""

import argparse
import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from helpers import ReplayBuffer
from models import DQN
from env import StockEnv


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
    device = torch.device("cuda")
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor
    device = torch.device("cpu")


class Agent:
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = 3

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        if random.random() > epsilon:
            state = torch.tensor(np.float32(state)).type(dtype)
            q_value = self.q_network.forward(state)
            return q_value.max(1)[1].data
        return torch.randint(low=0, high=3, size=(state.shape[0], 1)).squeeze(1)


def compute_td_loss(agent, batch_size, replay_buffer, optimizer, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).type(dtype)
    next_state = torch.tensor(np.float32(next_state)).type(dtype)
    action = torch.tensor(action).type(dtypelong)
    reward = torch.tensor(reward).type(dtype)
    done = torch.tensor(done).type(dtype)
    # Normal DDQN update
    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # double q-learning
    online_next_q_values = agent.q_network(next_state)
    _, max_indices = torch.max(online_next_q_values, dim=1)
    target_q_values = agent.target_q_network(next_state)
    next_q_value = torch.gather(target_q_values, 1, max_indices.unsqueeze(1))
    expected_q_value = reward + gamma * next_q_value.squeeze(1) * (1 - done)
    loss = (q_value - expected_q_value.data).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


def test(agent, start_date, end_date, params):
    test_env = StockEnv(ts_window=params.ts_window, start_date=start_date, end_date=end_date)



def run_gym(params):
    env = StockEnv(ts_window=30, start_date='20100101', end_date='20190101')
    q_network = DQN(num_stocks=params.ticker_num, num_days=params.ts_window)
    target_q_network = deepcopy(q_network)

    if USE_CUDA:
        q_network = q_network.to(device)
        target_q_network = target_q_network.to(device)

    agent = Agent(env, q_network, target_q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    replay_buffer = ReplayBuffer(params.replay_size)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()  # state shape: [num_stocks, ts_window]
    for ts in range(1, params.max_ts + 1):
        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action.cpu().numpy())
        reward *= params.reward_scale
        replay_buffer.push_batch(state, action, reward, next_state, done)

        state = info
        episode_reward += reward.mean()

        if done.all():
            env.stats()
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > params.start_train_ts and ts % params.update_every == 0:
            # Update the q-network & the target network
            for _ in range(params.gradient_step):
                loss = compute_td_loss(
                    agent, params.batch_size, replay_buffer, optimizer, params.gamma
                )
                losses.append(loss.data)

            if ts % params.target_network_update_f == 0:
                hard_update(agent.q_network, agent.target_q_network)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Reward: {}".format(all_rewards[-1])
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
            print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--start_train_ts", type=int, default=100)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=int, default=1000)
    parser.add_argument("--max_ts", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--target_network_update_f", type=int, default=1000)
    parser.add_argument("--ticker_num", type=int, default=10)
    parser.add_argument("--ts_window", type=int, default=30)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--gradient_step", type=int, default=1)
    run_gym(parser.parse_args())
