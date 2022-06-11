import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def compute_portvals(
        df_trades,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param df_trades: A data frame whose values represent trades for each day. Legal values are +1000.0 indicating a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are also legal when switching from long to short or short to long so long as net holdings are constrained to -1000, 0, and 1000. 
    :type df_trades: pandas.DataFrame
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    
    orders_df = df_trades
    # sort by date
    orders_df.sort_index(inplace=True)
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    dates = pd.date_range(start_date, end_date)

    symbols = sorted(list(df_trades.columns))

    # Notes: Leave addSPY default (True)
    # addSPY=False will give the wrong number of trading days
    prices_df = get_data(symbols, dates)
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')

    # remove SPY if not in list
    prices_df = prices_df[symbols]
    prices_df['Cash'] = 1.0

    trades_df = prices_df.copy()
    trades_df[symbols] = 0.0
    trades_df['Cash'] = 0.0

    for ticker in symbols:
        for date, row in orders_df.iterrows():
            shares = row[ticker]
            price = prices_df[ticker][date]
            trades_df.loc[date, ticker] += shares
            value = -1 * shares * price

            transaction_cost = commission + impact * price * shares
            trades_df.loc[date, 'Cash'] += value - transaction_cost

    holdings_df = trades_df.copy().cumsum()
    holdings_df['Cash'] = holdings_df['Cash'] + start_val

    values_df = prices_df * holdings_df
    portvals = values_df.sum(axis=1)

    return portvals

