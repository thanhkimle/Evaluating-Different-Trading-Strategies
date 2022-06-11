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

def compare_impacts(
    symbol='JPM',
    sd=dt.datetime(2008, 1, 1), 
    ed=dt.datetime(2009, 12, 31),
    sv=100000, 
    commission=0.0, 
    impacts=[0, 0.005, 0.05, 0.5], 
    verbose=False):

    if verbose:
        print()
        print(f'############################ EXPERIMENT 2 ############################')
        print(f"Date Range: {sd} to {ed} for {symbol}")
        print()

        
    fig, ax = plt.subplots(figsize=(12, 5))
    title = (f"Strategy Learner With Varying Impact Values ({symbol})")
    ax.set(title=title,
           xlabel='Date',
           ylabel="Normalized Portfolio Value")

    n = len(impacts)
    # colors = plt.cm.jet(np.linspace(0,1,n))
    count_trades = np.zeros([n])
    cum_rets = np.zeros([n])
    for i in range(n):
        learner = sl.StrategyLearner(impact=impacts[i])
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sl_trades_df = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

        sl_portvals = compute_portvals(sl_trades_df, start_val=sv, commission=commission, impact=impacts[i])
        sl_portvals_norm, sl_cum_ret, sl_avg_daily_ret, sl_std_daily_ret, sl_sharpe_ratio = ms.cal_portfolio_stats(sl_portvals)

        if verbose:
            print(f"Strategy Learner - Impact = {impacts[i]}")
            print(f"Cumulative Return: {round(sl_cum_ret,4)}")
            print(f"Average Daily Return: {round(sl_avg_daily_ret,4)}")
            print(f"Standard Deviation: {round(sl_std_daily_ret,4)}")
            print()
        
        ax.plot(sl_portvals_norm, label=(f"Impact {impacts[i]}"))

        # count = sl_trades_df[sl_trades_df != 0].count()[0]
        # count_trades.append(count)
        count_trades[i] = sl_trades_df[sl_trades_df != 0].count()[0]

        # cum_rets.append(sl_cum_ret*100)
        cum_rets[i] = sl_cum_ret*100

    ax.tick_params(axis='x', labelrotation=30)
    # ax.grid()
    ax.legend(loc='upper left', shadow=True, ncol=1)
    fig.tight_layout() 
    fig.savefig("impacts_1.png")
    # plt.show()

    # print(count_trades)
    # print(cum_rets)

    fig2, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(impacts, count_trades)
    ax1.set_title('Total Number of Trades vs. Impacts')
    ax1.set(xlabel='Impact Values', ylabel='Trades')
    ax1.grid()
    ax2.plot(impacts, cum_rets)
    ax2.set_title('Cumulative Return vs. Impacts')
    ax2.set(xlabel='Impact Values', ylabel='Percentage')
    ax2.grid()

    # ax2.grid()
    fig2.tight_layout() 
    fig2.savefig('impacts_2.png')

    plt.close()


