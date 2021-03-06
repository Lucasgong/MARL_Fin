#!/usr/bin/env python3
import argparse
import logging
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from env import StockEnv
from helpers import ReplayBuffer
from models import DQN

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    # dtype = torch.cuda.FloatTensor
    # dtypelong = torch.cuda.LongTensor
    device = torch.device("cuda")
else:
    print("NOT Using GPU: GPU not requested or not available.")
    # dtype = torch.FloatTensor
    # dtypelong = torch.LongTensor
    device = torch.device("cpu")


def get_logger(logpath,
               filepath,
               package_files=[],
               displaying=False,
               saving=True,
               debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Agent:
    def __init__(self, q_network, target_q_network, lr):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = 3
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        state = torch.tensor(np.float32(state)).to(device)
        if random.random() > epsilon:
            q_value = self.q_network.forward(state)
            return q_value.max(1)[1].data
        return torch.randint(low=0, high=3,
                             size=(state.shape[0], 1)).squeeze(1)

    def save(self, agent_id, save_dir):
        save_f = Path(f'{save_dir}/model/agent_{agent_id}.pth')
        save_f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), save_f)
        # print("Model has been saved...")

    def load(self, agent_id, save_dir):
        self.q_network.load_state_dict(
            torch.load(f'{save_dir}/model/agent_{agent_id}.pth'))
        self.target_q_network = deepcopy(self.q_network)
        # print("Model has been loaded...")


def corrcoef(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) *
                                 torch.sqrt(torch.sum(vy**2)))
    return corr


def compute_local_loss(agent, batch_size, replay_buffer, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).to(device)
    next_state = torch.tensor(np.float32(next_state)).to(device)
    action = torch.tensor(action).to(device)
    reward = torch.tensor(reward).to(device)
    done = torch.tensor(np.float32(done)).to(device)
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
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return loss


def compute_global_loss(agent_ls, batch_size, replay_buffer, global_optimizer,
                        global_loss_scale):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).to(device)
    q_values = [agent.q_network(state) for agent in agent_ls]
    eta_values = [q_value[:, -1] - q_value[:, 0] for q_value in q_values]

    q_target_values = [agent.target_q_network(state) for agent in agent_ls]
    eta_target_values = [
        q_target_value[:, -1] - q_target_value[:, 0]
        for q_target_value in q_target_values
    ]

    corrcoef_ls = [
        corrcoef(eta_values[i], eta_target_values[j])**2
        for i in range(len(eta_values)) for j in range(len(eta_target_values))
        if i != j
    ]

    global_loss = torch.tensor(0.0).to(device)
    for corr in corrcoef_ls:
        global_loss += corr
    global_loss *= global_loss_scale

    global_optimizer.zero_grad()
    global_loss.backward()
    global_optimizer.step()

    return global_loss


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay)


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(),
                              q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(),
                              q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


def test_agent(test_env, agent_ls, logger):
    state = test_env.reset()

    while True:
        for idx, agent in enumerate(agent_ls):
            action = agent.act(state, epsilon=0)
            _, _, done, _ = test_env.predict(action.cpu().numpy(),
                                             agent_id=idx)
        if done.all():
            logger.info('test')
            test_env.stats()
            break
        state = test_env.step()


def test_portfolio(train_env, test_env, agent_ls, logger):
    for env in [train_env, test_env]:
        state = env.reset()
        while True:
            with torch.no_grad():
                q_values = [
                    agent.q_network(
                        torch.as_tensor(state,
                                        device=device,
                                        dtype=torch.float32))
                    for agent in agent_ls
                ]
                eta_values = [
                    q_value[:, -1] - q_value[:, 0] for q_value in q_values
                ]
                eta_values = [
                    eta_value / eta_value.abs().sum()
                    for eta_value in eta_values
                ]

                [
                    env.predict(action.cpu().numpy(), agent_id=idx)
                    for idx, action in enumerate(eta_values)
                ]

                portfolio = torch.row_stack(eta_values).mean(axis=0)
                portfolio /= portfolio.abs().sum()
                portfolio += 1

                _, _, done, _ = env.predict(portfolio.cpu().numpy(),
                                            agent_id='portfolio')

            if done.all():
                logger.info(env.name)
                env.stats()
                break

            state = env.step()


def make_agent(params, hidden_dims=-1):
    if hidden_dims == -1:
        hidden_dims = params.hidden_dims
    q_network = DQN(num_days=params.ts_window, hidden_dims=hidden_dims)
    target_q_network = deepcopy(q_network)

    if USE_CUDA:
        q_network = q_network.to(device)
        target_q_network = target_q_network.to(device)

    agent = Agent(q_network, target_q_network, lr=params.learning_rate)

    return agent


def run_gym(params):
    logpath = Path(os.path.join(params.save_dir, 'logs'))
    logpath.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger(logpath=logpath, filepath=os.path.abspath(__file__))
    logger.info(params)

    env = StockEnv(ts_window=params.ts_window,
                   start_date='20100101',
                   end_date='20190101',
                   logger=logger,
                   agent_num=params.agent_num,
                   ic_coef=params.ic_scale,
                   name='train')

    train_env = deepcopy(env)
    test_env = StockEnv(ts_window=params.ts_window,
                        start_date='20190102',
                        end_date='20210603',
                        logger=logger,
                        agent_num=params.agent_num,
                        ic_coef=params.ic_scale,
                        name='test')

    agent_ls = [
        make_agent(params, hidden_dims=(2**(6 + i), 2**(6 + i)))
        for i in range(params.agent_num)
    ]
    
    [agent.load(idx,'experiments/v1_10d_10') for idx,agent in enumerate(agent_ls)]
    
    global_optimizer = optim.Adam([{
        'params': agent.q_network.parameters()
    } for agent in agent_ls],
                                  lr=params.learning_rate)

    replay_buffer = ReplayBuffer(params.replay_size)

    losses, all_rewards = [], []
    episode_reward = 0

    state = env.reset()
    # state shape: [num_stocks, ts_window]

    for ts in range(1, params.max_ts + 1):

        epsilon = get_epsilon(params.epsilon_start, params.epsilon_end,
                              params.epsilon_decay, ts)

        for idx, agent in enumerate(agent_ls):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.predict(action.cpu().numpy(),
                                                      agent_id=idx)
            reward *= params.reward_scale
            replay_buffer.push_batch(state, action, reward, next_state, done)
            episode_reward += reward.mean()

        if done.all():
            env.stats()
            state = env.reset()
            all_rewards.append(episode_reward / params.agent_num)
            episode_reward = 0

        state = env.step()

        if len(replay_buffer
               ) > params.start_train_ts and ts % params.update_every == 0:
            # Update the q-network & the target network
            for _ in range(params.gradient_step):
                all_loss = 0
                for agent in agent_ls:
                    loss = compute_local_loss(agent, params.batch_size,
                                              replay_buffer, params.gamma)
                    all_loss += loss.data

                global_loss = compute_global_loss(agent_ls, params.batch_size,
                                                  replay_buffer,
                                                  global_optimizer,
                                                  params.global_loss_scale)
                all_loss += global_loss.data
                losses.append(all_loss)

            if ts % params.target_network_update_f == 0:
                for agent in agent_ls:
                    hard_update(agent.q_network, agent.target_q_network)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Reward: {}".format(all_rewards[-1])
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
            logger.info(out_str)
            [
                agent.save(agent_id=i, save_dir=params.save_dir)
                for i, agent in enumerate(agent_ls)
            ]

            test_agent(test_env, agent_ls, logger)

            if ts > 50000:
                test_portfolio(train_env, test_env, agent_ls, logger)

            logger.info('------------------')


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
    parser.add_argument("--save_dir", type=str, default='experiments/v1')

    parser.add_argument("--ts_window", type=int, default=10)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--gradient_step", type=int, default=1)
    parser.add_argument("--hidden_dims", type=tuple, default=(256, 256))
    parser.add_argument("--agent_num", type=int, default=4)
    parser.add_argument("--ic_scale", type=float, default=10.0)
    parser.add_argument("--global_loss_scale", type=float, default=1)

    run_gym(parser.parse_args())
