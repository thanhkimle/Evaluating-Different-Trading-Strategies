import datetime as dt	  	   		   	 		  		  		    	 		 		   		 		  
import random 		  		    	 		 		   		 		  
import pandas as pd		  	   		   	 		  		  		    	 		 		   		 		  
import util as ut
import numpy as np
from marketsimcode import compute_portvals
import BagLearner as bl
import RTLearner as rtl  
import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt

def compare_strategies(
    benchmark_df,
    ms_df,
    sl_df,
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
    benchmark_portvals_norm, benchmark_cum_ret, benchmark_avg_daily_ret, benchmark_std_daily_ret, benchmark_sharpe_ratio = ms.cal_portfolio_stats(benchmark_portvals)
    # print(benchmark_portvals_norm.index)

    # Manual Strategy
    ms_portvals = compute_portvals(ms_df, start_val=sv, commission=commission, impact=impact)
    ms_portvals_norm, ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret, ms_sharpe_ratio = ms.cal_portfolio_stats(ms_portvals)

    # Strategy Learner
    sl_portvals = compute_portvals(sl_df, start_val=sv, commission=commission, impact=impact)
    sl_portvals_norm, sl_cum_ret, sl_avg_daily_ret, sl_std_daily_ret, sl_sharpe_ratio = ms.cal_portfolio_stats(sl_portvals)

    if verbose:
        print()
        # print(f'############################ EXPERIMENT 1 ############################')
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
        print("Strategy Learner")
        print(f"Cumulative Return: {round(sl_cum_ret,4)}")
        print(f"Average Daily Return: {round(sl_avg_daily_ret,4)}")
        print(f"Standard Deviation: {round(sl_std_daily_ret,4)}")
        print()

    # Benchmark Strategy vs. Manual Strategy vs. Strategy Learner Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    title = (f"Benchmark Strategy vs. Manual Strategy vs. Strategy Learner ({symbol})")
    ax.set(title=title,
           xlabel='Date',
           ylabel="Normalized Portfolio Value")
    ax.plot(benchmark_portvals_norm, "green", label="Benchmark")
    ax.plot(ms_portvals_norm, "red", label='Manual Strategy')
    ax.plot(sl_portvals_norm, "blue", label='Strategy Learner')

    ax.tick_params(axis='x', labelrotation=30)
    # ax.grid()
    ax.legend(loc='upper center', shadow=True, ncol=4)
    fig.savefig(fig_name)
    # plt.show()
    plt.close()
