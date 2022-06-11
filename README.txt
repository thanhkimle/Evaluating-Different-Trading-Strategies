Libraries used with Python3:
- datetime  	   		   	 		  		  		    	 		 		   		 		  
- random 		  		    	 		 		   		 		  
- pandas	  	   		   	 		  		  		    	 		 		   		 		  
- numpy 
- matplotlib
- scipy

Use the following command to install these libraries:
$ pip install <lib name>
i.e.
$ pip install scipy

=================================================================================================

Invoked and run the code by using the following command:
$ PYTHONPATH=../:. python testproject.py

=================================================================================================

1. indicators.py
Code implementing indicators as functions that operate on DataFrames.
The three technical indicators used in this project are:
1. Bollinger Bands Percentage (BBP)
2. Momentum
3. Mean Average Convergence Divergence (MACD)

2. marketsimcode.py
The function compute_portvals() accepts a “trades” DataFrame.
The result (portvals) is a single-column dataframe, containing the value of the portfolio 
for each trading day in the first column from start_date to end_date, inclusive.

3. ManualStrategy.py
This file contain code to normalize prices, generate benchmark portfolio, manual strategy 
testPolicy, calculate portfolio statistic, and compare strategies.

4. StrategyLearner.py
A strategy learner that can learn a trading policy using the same indicators used in 
ManualStrategy. This class has the following method: add_evidence and testPolicy		  	   		   	 		  		  		    	 		 		   		 	

5. DTLearner.py
This is a Decision Tree Learner.

6. RTLearner.py
This is a  Decision Tree Learner with the choice of feature to split is made randomly. 
RTLearner class inherit from DTLearner class.

7. BagLearner.py
Bootstrap Aggregation as a Python class named BagLearner.
This API is designed so that the BagLearner can accept any learner
(e.g., RTLearner, LinRegLearner, even another BagLearner) as input
and use it to generate a learner ensemble.

8. experiment1.py
Code for experiment 1 that compare benchmark, manual strategy, and strategy learner.

9. Experiment2.py
Code for experiment 2 that compare varying impact values for strategy learner.

10. testproject.py
This file is the entry point to the project. Invoked this file will generate all the plots 
in the report.

11. util.py
Use the get_data function to read stock data
