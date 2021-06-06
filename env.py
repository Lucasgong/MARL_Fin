import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StockEnv:
    def __init__(self, ts_window, start_date='20100101', end_date='20190101', lookbacks=-1):
        self.ts_window = ts_window
        lookbacks = self.ts_window if lookbacks < 0 else lookbacks
        self.__init_market_data(start_date, end_date, lookbacks)
        self.reset()

    def reset(self):
        self.today = self.ts_window
        self.infos = []
        self.pnl = pd.Series(dtype=np.float64)
        self.gen_valid_stocks()

        return self.valid_states

    def gen_valid_stocks(self):
        state = self.univ.iloc[self.today -
                               self.ts_window:self.today, :].copy()
        valid = self.valid.iloc[self.today-1,
                                :] & (state.notna().sum() == self.ts_window)
        state = state.loc[:, valid]
        self.valid_states = state.values.T
        self.valid_stocks = state.columns

    def step(self, nonnegative_action):
        '''
        nonnegative_action \in [0,1,2]
        '''
        assert nonnegative_action.shape == (len(self.valid_stocks),)
        actions = nonnegative_action - 1
        self.today += 1

        next_rets = self.univ.iloc[self.today, :][self.valid_stocks].values
        done = np.array([pd.isna(ret) for ret in next_rets])
        if self.today == (self.endday - 1):
            done = np.array([True]*len(done))
        next_rets = np.nan_to_num(next_rets, 0)

        if actions.std() == 0:
            ic = 0
        else:
            ic = np.corrcoef(next_rets, actions)[0, 1]

        rewards = next_rets*actions
        pnl = np.mean(rewards)
        next_states = np.concatenate(
            [self.valid_states[:, 1:], next_rets[:, np.newaxis]], axis=1)

        self.gen_valid_stocks()

        tradeday = self.tradedays[self.today]
        info = self.valid_states
        self.pnl[tradeday] = pnl

        return next_states, rewards, done, info

    def plot_pnl(self):
        self.pnl.cumsum().plot()
        plt.show()

    def stats(self):
        ann_pnl = self.pnl.mean()*252
        sharpe = (self.pnl.mean()-0)/(self.pnl.std()+1e-8)*np.sqrt(252)
        print(f'annret:{ann_pnl},sharpe:{sharpe}')

    def __init_market_data(self, start_date, end_date, lookbacks):

        valid = pd.read_csv('data/top1500.csv', index_col=0,
                            sep='|', parse_dates=True)

        start_idx = valid.index.searchsorted(start_date)
        start_idx = max(start_idx-lookbacks, 0)
        end_idx = valid.index.searchsorted(end_date)
        end_idx = end_idx + \
            1 if valid.index[end_idx].strftime(
                '%Y%m%d') == end_date else end_idx

        valid = valid.iloc[start_idx:end_idx, :]

        ret = pd.read_csv('data/cps.csv', index_col=0, sep='|',
                          parse_dates=True).pct_change()*100
        ret = ret.iloc[start_idx:end_idx, :]
        ret.dropna(axis=1, how='all', inplace=True)
        ret.dropna(axis=0, how='all', inplace=True)

        self.tradedays = ret.index
        self.endday = len(self.tradedays)
        self.tickers = ret.columns
        self.valid = valid.loc[self.tradedays, self.tickers].astype(bool)
        self.univ = ret
        print(
            f'init env:firstday:{self.tradedays[self.ts_window].strftime("%Y%m%d")} tradedays:{len(self.tradedays)},tickers:{len(self.tickers)}')


if __name__ == '__main__':
    ts_window = 10
    env = StockEnv(ts_window, start_date='20100101', end_date='20190101')

    state = env.reset()
    #reward_ls = []
    while True:
        action = 2 * np.ones(state.shape[0])
        next_state, reward, terminated, state = env.step(action)

        if terminated.all():
            break
    env.stats()
