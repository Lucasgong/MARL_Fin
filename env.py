import logging
from os import name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StockEnv:
    def __init__(self,
                 ts_window,
                 logger,
                 start_date='20100101',
                 end_date='20190101',
                 lookbacks=-1,
                 ic_coef=10,
                 agent_num=1,
                 name='test'):

        self.name = name
        self.logger = logger
        self.ts_window = ts_window
        lookbacks = self.ts_window if lookbacks < 0 else lookbacks
        self.ic_coef = ic_coef
        self.agent_num = agent_num
        self.__init_market_data(start_date, end_date, lookbacks)
        self.reset()

    def reset(self):
        self.today = self.ts_window
        self.infos = []
        self.pnl = pd.DataFrame(dtype=np.float64,
                                columns=range(self.agent_num))
        self.gen_valid_info()

        return self.valid_states

    def gen_valid_info(self):
        state = self.univ.iloc[self.today -
                               self.ts_window:self.today, :].copy()
        valid = self.valid.iloc[self.today - 1, :] & (state.notna().sum()
                                                      == self.ts_window)
        state = state.loc[:, valid]
        self.valid_states = state.values.T
        self.valid_stocks = state.columns

        target_rets = self.univ.iloc[self.today +
                                     1, :][self.valid_stocks].values
        done = np.array([pd.isna(ret) for ret in target_rets])
        if self.today == (self.endday - 2):
            done = np.array([True] * len(done))
        target_rets = np.nan_to_num(target_rets, 0)

        today_ret = self.univ.iloc[self.today, :][self.valid_stocks].values
        today_ret = np.nan_to_num(today_ret, 0)

        self.next_states = np.concatenate(
            [self.valid_states[:, 1:], today_ret[:, np.newaxis]], axis=1)
        self.done = done
        self.target_rets = target_rets

    def predict(self, nonnegative_action, agent_id=0):
        assert nonnegative_action.shape == (len(self.valid_stocks), )

        actions = nonnegative_action - 1

        if actions.std() == 0:
            ic = 0
        else:
            ic = np.corrcoef(self.target_rets, actions)[0, 1]

        rewards = self.target_rets * actions
        pnl = rewards.sum() / (1e-8 + abs(actions).sum())
        tradeday = self.tradedays[self.today + 1]
        self.pnl.loc[pd.to_datetime(tradeday, format='%Y%m%d'), agent_id] = pnl

        rewards += self.ic_coef * ic

        return self.next_states, rewards, self.done, []

    def step(self):
        self.today += 1
        self.gen_valid_info()

        return self.valid_states

    def render(self):
        self.pnl.cumsum().plot()
        plt.show()

    def stats(self):
        ann_pnl = self.pnl.mean() * 252
        sharpe = (self.pnl.mean() - 0) / (self.pnl.std() + 1e-8) * np.sqrt(252)
        for i, name in enumerate(self.pnl.columns):
            self.logger.info(
                f'{name}:  annret:{ann_pnl[name]:.3f},  sharpe:{sharpe[name]:.3f}'
            )
        self.logger.info(self.pnl.corr())

    def __init_market_data(self, start_date, end_date, lookbacks):

        start_date = int(start_date)
        end_date = int(end_date)

        valid = pd.read_csv('data/top1500.csv', index_col=0, sep='|')

        start_idx = valid.index.searchsorted(start_date)
        start_idx = max(start_idx - lookbacks, 0)
        end_idx = valid.index.searchsorted(end_date)
        end_idx = end_idx + 1 if valid.index[end_idx] == end_date else end_idx

        valid = valid.iloc[start_idx:end_idx, :]

        ret = pd.read_csv('data/cps.csv', index_col=0,
                          sep='|').pct_change() * 100
        ret = ret.iloc[start_idx:end_idx, :]
        ret.dropna(axis=1, how='all', inplace=True)
        ret.dropna(axis=0, how='all', inplace=True)

        self.tradedays = ret.index
        self.endday = len(self.tradedays)
        self.tickers = ret.columns
        self.valid = valid.loc[self.tradedays, self.tickers].astype(bool)
        self.univ = ret
        self.logger.info(
            f'init env:firstday:{self.tradedays[self.ts_window]} tradedays:{len(self.tradedays)},tickers:{len(self.tickers)}'
        )


if __name__ == '__main__':

    logger = logging.getLogger()

    level = logging.INFO
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    ts_window = 10
    agent_num = 4
    env = StockEnv(ts_window,
                   logger=logger,
                   start_date='20120101',
                   end_date='20190101',
                   agent_num=agent_num)

    state = env.reset()
    #reward_ls = []
    while True:
        for idx in range(agent_num):
            action = 2 * np.ones(state.shape[0])
            next_state, reward, terminated, _ = env.predict(action, idx)
        if terminated.all():
            break
        state = env.step()

    env.stats()
    #env.render()
