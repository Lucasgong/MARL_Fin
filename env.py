import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StockEnv:
    def __init__(self, ts_window, start_date='20100101', end_date='20190101', lookbacks=-1, ic_coef=10):
        self.ts_window = ts_window
        lookbacks = self.ts_window if lookbacks < 0 else lookbacks
        self.ic_coef = ic_coef
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

        target_rets = self.univ.iloc[self.today+1, :][self.valid_stocks].values
        done = np.array([pd.isna(ret) for ret in target_rets])
        if self.today == (self.endday - 2):
            done = np.array([True]*len(done))
        target_rets = np.nan_to_num(target_rets, 0)

        if actions.std() == 0:
            ic = 0
        else:
            ic = np.corrcoef(target_rets, actions)[0, 1]

        rewards = target_rets*actions + self.ic_coef * ic
        pnl = np.mean(rewards)

        today_ret = self.univ.iloc[self.today, :][self.valid_stocks].values
        today_ret = np.nan_to_num(today_ret, 0)
        next_states = np.concatenate(
            [self.valid_states[:, 1:], today_ret[:, np.newaxis]], axis=1)

        self.today += 1
        self.gen_valid_stocks()

        tradeday = self.tradedays[self.today]
        info = self.valid_states
        self.pnl[pd.to_datetime(tradeday, format='%Y%m%d')] = pnl

        return next_states, rewards, done, info

    def render(self):
        self.pnl.cumsum().plot()
        plt.show()

    def stats(self):
        ann_pnl = self.pnl.mean()*252
        sharpe = (self.pnl.mean()-0)/(self.pnl.std()+1e-8)*np.sqrt(252)
        print(f'annret:{ann_pnl},sharpe:{sharpe}')

    def __init_market_data(self, start_date, end_date, lookbacks):

        start_date = int(start_date)
        end_date = int(end_date)

        valid = pd.read_csv('data/top1500.csv', index_col=0, sep='|')

        start_idx = valid.index.searchsorted(start_date)
        start_idx = max(start_idx-lookbacks, 0)
        end_idx = valid.index.searchsorted(end_date)
        end_idx = end_idx + 1 if valid.index[end_idx] == end_date else end_idx

        valid = valid.iloc[start_idx:end_idx, :]

        ret = pd.read_csv('data/cps.csv', index_col=0,
                          sep='|').pct_change()*100
        ret = ret.iloc[start_idx:end_idx, :]
        ret.dropna(axis=1, how='all', inplace=True)
        ret.dropna(axis=0, how='all', inplace=True)

        self.tradedays = ret.index
        self.endday = len(self.tradedays)
        self.tickers = ret.columns
        self.valid = valid.loc[self.tradedays, self.tickers].astype(bool)
        self.univ = ret
        print(
            f'init env:firstday:{self.tradedays[self.ts_window]} tradedays:{len(self.tradedays)},tickers:{len(self.tickers)}')


if __name__ == '__main__':
    ts_window = 100
    env = StockEnv(ts_window, start_date='20120101', end_date='20190101')

    state = env.reset()
    #reward_ls = []
    while True:
        action = 2 * np.ones(state.shape[0])
        next_state, reward, terminated, state = env.step(action)

        if terminated.all():
            break
    env.stats()
    env.render()
