import datetime as dt
import pandas as pd
import ManualStrategy as ms
import StrategyLearner as sl
import experiment1 as e1
import experiment2 as e2
import warnings
import numpy as np


if __name__ == "__main__":

    # pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # GT ID
    np.random.seed(903052835)

    symbol = 'JPM'
    # symbols = ['JPM', 'ML4T-220', 'AAPL', 'SINE_FAST_NOISE', 'UNH']

    sv=100000
    commission=9.95
    impact=0.005
    verbose = True

    # commission=0
    # impact=0

    ############################ IN SAMPLE ############################

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)

    # Manual Strategy's Trade DataFrame 
    ms_trades_df = ms.testPolicy(
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed)

    # Benchmark's Trade DataFrame 
    benchmark_trades_df = ms.get_benchmark(
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed)

    # Compare Manual Strategy vs benchmark
    ms.compare_strategies(
        benchmark_trades_df,
        ms_trades_df, 
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed, 
        sv=sv,
        fig_name='ms_in_sample.png', 
        verbose=verbose)

    # Strategy Learner's Trade DataFrame 
    learner = sl.StrategyLearner()
    learner.add_evidence(
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed, 
        sv=sv)
    sl_trades_df = learner.testPolicy(
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed, 
        sv=sv)

    # Experiment 1
    e1.compare_strategies(
        benchmark_trades_df,
        ms_trades_df,
        sl_trades_df, 
        symbol=symbol, 
        sd=in_sample_sd, 
        ed=in_sample_ed,
        sv=sv, 
        fig_name='sl_in_sample.png', 
        verbose=verbose)

    # Experiment 2
    e2.compare_impacts(
        symbol=symbol,
        sd=in_sample_sd, 
        ed=in_sample_ed,
        sv=sv, 
        commission=0, 
        impacts=[0.001, 0.008, 0.025, 0.055, 0.155, 0.555, 0.955], 
        verbose=verbose)

    ############################ OUT SAMPLE ############################

    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)

    # Manual Strategy's Trade DataFrame 
    ms_trades_df = ms.testPolicy(
        symbol=symbol, 
        sd=out_sample_sd, 
        ed=out_sample_ed)

    # Benchmark's Trade DataFrame 
    benchmark_trades_df = ms.get_benchmark(
        symbol=symbol, 
        sd=out_sample_sd, 
        ed=out_sample_ed)

    # Compare Manual Strategy vs benchmark
    ms.compare_strategies(
        benchmark_trades_df,
        ms_trades_df, 
        symbol=symbol, 
        sd=out_sample_sd, 
        ed=out_sample_ed, 
        sv=sv,
        fig_name='ms_out_sample.png', 
        verbose=verbose)

    # Strategy Learner's Trade DataFrame 
    sl_trades_df = learner.testPolicy(
        symbol=symbol, 
        sd=out_sample_sd, 
        ed=out_sample_ed, 
        sv=sv)

    # Experiment 1
    e1.compare_strategies(
        benchmark_trades_df,
        ms_trades_df,
        sl_trades_df, 
        symbol=symbol, 
        sd=out_sample_sd, 
        ed=out_sample_ed,
        sv=sv, 
        fig_name='sl_out_sample.png', 
        verbose=verbose)    

    print('Completed')

    