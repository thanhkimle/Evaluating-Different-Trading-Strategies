import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data


# Indicator 1 - Simple Moving Average
def get_sma(price, lookback):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    price_per_sma = price / sma
    return sma, price_per_sma


def plot_sma(price):
    sma, price_per_sma = get_sma(price, 20)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set(title='Simple Moving Average (20-day SMA)',
           xlabel='Date',
           ylabel='Price')
    ax.plot(price, label='Normalized Price')
    ax.plot(sma, label='SMA')
    ax.plot(price_per_sma, label='Price/SMA')
    ax.grid(True)
    ax.legend()
    fig.savefig('Indicator1_SMA.png')
    plt.close()


# Indicator 2 - Bollinger Bands
def get_bb(price, lookback):
    rolling_mean = price.rolling(window=20, min_periods=lookback).mean()
    rolling_std = price.rolling(window=20, min_periods=lookback).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bbp = (price - lower_band) / (upper_band - lower_band)
    return bbp, upper_band, lower_band, rolling_mean


def plot_bb(price):
    bbp, upper_band, lower_band, sma = get_bb(price, 20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.5)
    # fig, ax = plt.subplots(figsize=(20, 8))
    ax1.set(title='Bollinger Bands (20-day SMA)',
            xlabel='Date',
            ylabel='Price')
    ax1.plot(price, label='Normalized Price')
    ax1.plot(sma, label='SMA')
    ax1.plot(upper_band, label='Top Band')
    ax1.plot(lower_band, label='Lower Band')
    ax1.grid(True)
    ax1.legend()

    ax2.set(title='Bollinger Bands Percentage (20-day SMA)',
            xlabel='Date',
            ylabel='Percentage')
    ax2.plot(bbp, label='BBP')
    ax2.hlines(y=0.0, xmin=price.index.min(), xmax=price.index.max(), linestyles='--', color='r')
    ax2.hlines(y=1.0, xmin=price.index.min(), xmax=price.index.max(), linestyles='--', color='r')
    ax2.grid(True)
    ax2.legend()

    fig.savefig('Indicator2_BB.png')
    plt.close()


# Indicator 3 - Momentum
def get_momentum(price, lookback):
    momentum = price / price.shift(lookback) - 1
    return momentum


def plot_momentum(price):
    momentum = get_momentum(price, 20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    fig.subplots_adjust(hspace=0.5)
    ax1.set(title='Momentum',
            xlabel='Date',
            ylabel='Price')
    ax1.plot(price, label='Normalized Price')
    ax1.grid(True)
    ax1.legend()

    ax2.set(xlabel='Date',
            ylabel='Score')
    ax2.plot(momentum, label='Momentum (20-day)')
    ax2.grid(True)
    ax2.legend()

    fig.savefig('Indicator3_Momentum.png')
    plt.close()


# Indicator 4  - Exponential moving average (EMA)
def get_ema(price, days):
    ema = price.ewm(span=days, adjust=False).mean()
    return ema


def plot_ema(price):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.5)
    ax1.set(title='Exponential Moving Average (Short-term)',
            xlabel='Date',
            ylabel='Price')
    ax1.plot(price, label='Normalized Price')
    ax1.plot(get_ema(price, 12), label='12-day EMA')
    ax1.plot(get_ema(price, 26), label='26-day EMA')
    ax1.grid(True)
    ax1.legend()

    ax2.set(title='Exponential Moving Average (Long-term)',
            xlabel='Date',
            ylabel='Price')
    ax2.plot(price, label='Normalized Price')
    ax2.plot(get_ema(price, 50), label='50-day EMA')
    ax2.plot(get_ema(price, 200), label='200-day EMA')
    ax2.grid(True)
    ax2.legend()

    fig.savefig('Indicator4_EMA.png')
    plt.close()


# Indicator 5  - Mean Average Convergence Divergence
def get_macd(price):
    # Pandas DataFrame provides a function ewm(),
    # which together with the mean-function can calculate the
    # Exponential Moving Averages.
    ema_short = price.ewm(span=12, adjust=False).mean()
    ema_long = price.ewm(span=26, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def plot_macd(price):
    macd, signal = get_macd(price)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.5)
    ax1.set(title='Mean Average Convergence Divergence',
            xlabel='Date',
            ylabel='Price')
    ax1.plot(price, label='Normalized Price')
    ax1.grid(True)
    ax1.legend()

    ax2.set(xlabel='Date',
            ylabel='MACD Value')
    ax2.plot(macd, label='Slower Signal (EMA 12-days and 26-days window)')
    ax2.plot(signal, label='Faster Signal (9 day EMA of MACD)')
    ax2.grid(True)
    ax2.legend()

    fig.savefig('Indicator5_MACD.png')
    plt.close()


# def gen_report(symbol, sd, ed):
#     price = get_norm_prices(symbol, sd, ed)

#     plot_sma(price)
#     plot_bb(price)
#     plot_momentum(price)
#     plot_ema(price)
#     plot_macd(price)
