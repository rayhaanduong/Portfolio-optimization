from risk import volatility, sharpe_ratio, max_drawdown, sortino_ratio, cvar

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from scipy.stats import skew, kurtosis


def plot_portfolio(rebalance_output, title = "Portfolio Performance", benchmark: Optional[pd.Series] = None, plot : Optional[bool] = True):
    """
    Plot the performance and metrics of a PortfolioManager output.

    Parameters : 

        rebalance_output : dict
            Output from pm.run_monthly_rebalance or run_monthly_1overN
            Must contain keys:
                'weights', 'daily_returns', 'cumulative_returns',
                'max_drawdown', 'annualized_sharpe'
        title : str
            Title of the figure
            
        benchmark : Optional[pd.Series] 
        
    """
    weights = rebalance_output['weights']
    daily_ret = rebalance_output['daily_returns']
    cum_ret = rebalance_output['cumulative_returns']
    sharpe = rebalance_output['annualized_sharpe']
    max_dd = rebalance_output['max_drawdown']
    volatility_ = rebalance_output['volatility']
    sortino = rebalance_output["sortino"]
    cvar_ = rebalance_output["cvar"]
    
    
    print("Portfolio Metrics:")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Volatility  : {volatility_:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  cvar: {cvar_:.3f}")

    

    # Cumulative returns
    
    if plot == True :
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(cum_ret.index, cum_ret.values, color='tab:blue', lw = 1, label = 'Cumulative Returns', alpha = 0.8)
    
   
    
    
    if benchmark is not None:
        benchmark_aligned = benchmark.reindex(cum_ret.index).ffill()  # forward fill missing dates
        
        # Metrics
        
        daily_ret_bench = benchmark_aligned.pct_change().dropna()
        sharpe_bench = sharpe_ratio(daily_ret_bench)
        vol_bench = volatility(daily_ret_bench)
        cum = (1 + daily_ret_bench).cumprod()
        max_dd_bench = max_drawdown(cum)
        sortino_bench = sortino_ratio(daily_ret_bench)
        cvar_bench = cvar(daily_ret_bench)
        
        if plot == True: 
            axes.plot(cum.index, cum, color='tab:orange', lw = 1, linestyle='--', label='Benchmark', alpha = 0.8)
        
        
        print("\nBenchmark Metrics:")
        print(f"  Sharpe Ratio: {sharpe_bench:.2f}")
        print(f"  Volatility  : {vol_bench:.2f}")
        print(f"  Max Drawdown: {max_dd_bench:.2%}")
        print(f"  Sortino: {sortino_bench:.2f}")
        print(f"  cvar: {cvar_bench:.3f}\n")
        
        
            
        
    if plot == True:   
        axes.set_ylabel("Cumulative Returns")
        axes.set_title(f"{title}\nSharpe: {sharpe:.2f}, Max Drawdown: {max_dd:.2%}, Volatility: {volatility_:.2f}")
        axes.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    
    


def plot_return_distribution_with_stats(strategy_returns, benchmark_returns, bins=50, title = "Return Distribution"):
    """
    Plot histogram of daily returns for strategy vs benchmark with skewness and kurtosis.

    Parameters:
        strategy_returns : pd.Series
        benchmark_returns : pd.Series
        bins : int, number of histogram bins
        title : str
    """
    # Compute stats
    
    skew_strat = skew(strategy_returns)
    kurt_strat = kurtosis(strategy_returns)
    
    skew_bench = skew(benchmark_returns)
    kurt_bench = kurtosis(benchmark_returns)
    
    plt.figure(figsize=(10,6))
    
    plt.hist(strategy_returns, bins=bins, alpha=0.6, label=f"Strategy\nSkew={skew_strat:.2f}, Kurt={kurt_strat:.2f}", color='tab:blue', density=True)
    plt.hist(benchmark_returns, bins=bins, alpha=0.6, label=f"Benchmark\nSkew={skew_bench:.2f}, Kurt={kurt_bench:.2f}", color='tab:orange', density=True)
    
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()