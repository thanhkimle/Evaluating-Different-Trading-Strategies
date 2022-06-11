import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import get_macd, get_bb, get_momentum


def get_norm_prices(symbols, start_date, end_date):
    prices_df = get_data([symbols], pd.date_range(start_date, end_date))

    # drop SPY if not in list of symbols
    if 'SPY' not in symbols:
        prices_df.drop('SPY', axis=1, inplace=True)
    # prices_df = prices_df[symbols]

    # fill missing data forward than backward
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')

    prices_norm = prices_df / prices_df.iloc[0]

    return prices_norm


def get_benchmark(
    symbol='JPM', 
    sd=dt.datetime(2008, 1, 1), 
    ed=dt.datetime(2009, 12, 31), 
    shares=1000):

    # load the prices of the stocks during the investment period
    prices = get_norm_prices(symbol, sd, ed)

    # get the valid trading dates within the investment period
    dates = [prices.index[0], prices.index[-1]]

    # create the orders dataframe
    trades_df = pd.DataFrame(index=prices.index)
    trades_df[symbol] = 0
    trades_df[symbol].loc[prices.index[0]] = shares
    # trades_df[symbol].loc[prices.index[-1]] = -1 * shares

    return trades_df


def testPolicy(
    symbol='JPM', 
    sd=dt.datetime(2008, 1, 1), 
    ed=dt.datetime(2009, 12, 31), 
    sv=100000):

    # load the prices of the stocks during the investment period
    prices = get_norm_prices(symbol, sd, ed)

    # indicators
    lookback_win = 18
    bbp = get_bb(prices, lookback_win)[0]
    # print(bbp[symbol].iloc[20])
    macd, signal = get_macd(prices)
    momentum = get_momentum(prices, lookback_win) 
    # print(momentum)

    trades_df = pd.DataFrame(index=prices.index)
    trades_df[symbol] = 0

    current_pos = 0
    n = len(prices)
    for i in range(n):

        # BUY
        if bbp[symbol].iloc[i] < 0.20 \
            and momentum[symbol].iloc[i] < 0 \
            and macd[symbol].iloc[i] < signal[symbol].iloc[i]:            
            shares = 1000 - current_pos
            trades_df[symbol].loc[prices.index[i]] = shares
            current_pos = current_pos + shares
        
        # SELL
        elif bbp[symbol].iloc[i] > 0.80 \
            and momentum[symbol].iloc[i] > 0 \
            and macd[symbol].iloc[i] > signal[symbol].iloc[i]: 
            shares = -1000 - current_pos
            trades_df[symbol].loc[prices.index[i]] = shares
            current_pos = current_pos + shares

    return trades_df

def cal_portfolio_stats(portvals):
    portvals_norm = portvals / portvals.iloc[0]
    daily_returns = portvals_norm.copy()
    daily_returns[1:] = (portvals_norm[1:] / portvals_norm[:-1].values) - 1
    daily_returns = daily_returns[1:]

    cum_ret = (portvals_norm[-1] / portvals_norm[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252.0) * (avg_daily_ret / std_daily_ret)

    return portvals_norm, cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def compare_strategies(
    benchmark_df,
    ms_df,
    symbol='JPM',
    sd=dt.datetime(2008, 1, 1), 
    ed=dt.datetime(2009, 12, 31),
    sv=100000, 
    commission=9.95, 
    impact=0.005, 
    fig_name='compare_strategies.png', 
    verbose=False):

    # Benchmark Strategy
    benchmark_portvals = compute_portvals(benchmark_df, start_val=sv, commission=commission, impact=impact)
    benchmark_portvals_norm, benchmark_cum_ret, benchmark_avg_daily_ret, benchmark_std_daily_ret, benchmark_sharpe_ratio = cal_portfolio_stats(benchmark_portvals)
    # print(benchmark_portvals_norm.index)

    # Manual Strategy
    ms_portvals = compute_portvals(ms_df, start_val=sv, commission=commission, impact=impact)
    ms_portvals_norm, ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret, ms_sharpe_ratio = cal_portfolio_stats(ms_portvals)

    if verbose:
        print()
        print(f'############################ {fig_name} ############################')
        print(f"Date Range: {sd} to {ed} for {symbol}")
        print()
        print("Benchmark Strategy")
        print(f"Cumulative Return: {round(benchmark_cum_ret,4)}")
        print(f"Average Daily Return: {round(benchmark_avg_daily_ret,4)}")
        print(f"Standard Deviation: {round(benchmark_std_daily_ret,4)}")
        print()
        print("Manual Strategy")
        print(f"Cumulative Return: {round(ms_cum_ret,4)}")
        print(f"Average Daily Return: {round(ms_avg_daily_ret,4)}")
        print(f"Standard Deviation: {round(ms_std_daily_ret,4)}")
        print()

    # Manual vs Benchmark Strategy Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    title = (f"Manual vs Benchmark Strategy ({symbol})")
    ax.set(title=title,
           xlabel='Date',
           ylabel="Normalized Portfolio Value")
    ax.plot(benchmark_portvals_norm, "green", label="Benchmark")
    ax.plot(ms_portvals_norm, "red", label='Manual Strategy')

    flag_long = 0
    flag_short = 0

    n = len(ms_df)
    for i in range(n):
        if ms_df[symbol].iloc[i] > 0 and flag_long == 0:
            ax.axvline(x=ms_df.index[i], ls='--', color='blue', linewidth=0.5, label='LONG')
            flag_long = 1

        if ms_df[symbol].iloc[i] > 0 and flag_long == 1:
            ax.axvline(x=ms_df.index[i], ls='--', color='blue', linewidth=0.5)

        if ms_df[symbol].iloc[i] < 0 and flag_short == 0:
            ax.axvline(x=ms_df.index[i], ls='--', color='black', label='SHORT', linewidth=0.5)
            flag_short = 1

        if ms_df[symbol].iloc[i] < 0 and flag_short == 1:
            ax.axvline(x=ms_df.index[i], ls='--', color='black', linewidth=0.5)


    ax.tick_params(axis='x', labelrotation=30)
    # ax.grid()
    ax.legend(loc='upper left', shadow=True, ncol=1)
    fig.savefig(fig_name)
    # plt.show()
    plt.close()
