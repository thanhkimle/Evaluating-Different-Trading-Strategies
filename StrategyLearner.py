import datetime as dt	  	   		   	 		  		  		    	 		 		   		 		  
import random 		  		    	 		 		   		 		  
import pandas as pd		  	   		   	 		  		  		    	 		 		   		 		  
import util as ut
import numpy as np
from indicators import get_macd, get_bb, get_momentum
import BagLearner as bl
import RTLearner as rtl  
from ManualStrategy import get_norm_prices  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission  	

        # lookback window for indicators
        self.lookback_win = 8

        # N day return
        self.N = 10

        self.YSELL = 0.02 + self.impact
        self.YBUY = -0.02 - self.impact

        self.leaf_size = 8
        self.nbags = 30

        self.learner = bl.BagLearner(
            learner=rtl.RTLearner, 
            kwargs={"leaf_size": self.leaf_size }, 
            bags=self.nbags, 
            boost=False,                         
            verbose=False)	  	   		   	 		  		  		    	 		 		   		 		  

    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  

        prices = get_norm_prices(symbol, sd, ed)

        # indicators
        bbp = get_bb(prices, self.lookback_win)[0]
        macd, signal = get_macd(prices)
        momentum = get_momentum(prices, self.lookback_win) 

        indicators_df = pd.concat((bbp, momentum, macd, signal), axis=1)
        indicators_df = indicators_df.fillna(method='ffill').fillna(method='bfill')
        x_train = indicators_df[:-self.N].values

        y_train = np.zeros(x_train.shape[0])
        n = y_train.shape[0]
        for i in range(n):
            t = prices.index[i]
            t_N = prices.index[i + self.N]
            ret = (prices[symbol].loc[t_N]/prices[symbol].loc[t]) - 1

            if ret > self.YBUY:
                y_train[i] = 1 # LONG
            elif ret < self.YSELL:
                y_train[i] = -1 # SHORT
            else:
                y_train[i] = 0 # CASH

        self.learner.add_evidence(x_train, y_train)	   	 		  		  		    	 		 		   		 		  
                                                                                                
    # this method should use the existing policy and test it against new data  		  	   		   	 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        prices = get_norm_prices(symbol, sd, ed)

        # indicators
        bbp = get_bb(prices, self.lookback_win)[0]
        macd, signal = get_macd(prices)
        momentum = get_momentum(prices, self.lookback_win) 

        indicators_df = pd.concat((bbp, momentum, macd, signal), axis=1)
        indicators_df = indicators_df.fillna(method='ffill').fillna(method='bfill')
        x_test = indicators_df[:].values
        y_test = self.learner.query(x_test)

        trades_df = prices.copy()
        trades_df[symbol] = 0

        current_pos = 0
        n = len(prices)
        for i in range(n):
            # BUY
            if y_test[i] > 0:            
                shares = 1000 - current_pos
                trades_df[symbol].loc[prices.index[i]] = shares
                current_pos = current_pos + shares
            
            # SELL
            elif y_test[i] < 0: 
                shares = -1000 - current_pos
                trades_df[symbol].loc[prices.index[i]] = shares
                current_pos = current_pos + shares

        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(type(trades_df))  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(trades_df)  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices)  	

        return trades_df  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    pass
