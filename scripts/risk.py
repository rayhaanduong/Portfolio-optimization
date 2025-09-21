import numpy as np
import pandas as pd


def max_drawdown(cum_returns : pd.Series) -> float:

    """
    Compute the maximum drawdown of cumulative returns series (note that we want cumulative returns not log returns)

    Parameters :
        cum_return : pd.Series
    Returns :
        float : the maximum drawdown

    """
    running_max = cum_returns.cummax()
    return ((cum_returns - running_max) / running_max).min()


def volatility(returns : pd.Series, periods_per_year : int = 252) -> float:
    """
    Annualized volatility using sample std. We assume the data are daily, otherwise one can modify periods_per_year accordingly
    We always consider n - 1 ddof
    
    Parameters : 
        returns : pd.Series
    Returns:
        float : the volatility
    
    """
    
    return returns.std(ddof = 1) * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio using arithmetic mean
    
    Parameters : 
        returns : pd.Series
        periods_per_year : int : defaults to 252
        
    Returns :
        float : the Sharpe Ratio
    """
    
    return returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute the annualized Sortino ratio.

    Parameters:
        returns : pd.Series : daily portfolio returns
        periods_per_year : int : for annualization (default 252 for daily)
        
    Returns:
        float : annualized Sortino ratio
    """
    
    downside = returns[returns < 0]
    sigma_d = np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)
    mean_ret = returns.mean() * periods_per_year
    return mean_ret / sigma_d if sigma_d != 0 else np.nan


def cvar(returns, alpha=0.05):
    """
    Conditional VaR at confidence level alpha
    returns : pd.Series or np.array
    alpha : float, e.g., 0.05 for 5%
    """
    sorted_returns = np.sort(returns)
    index = int(np.floor(alpha * len(sorted_returns)))
    cvar_value = sorted_returns[:index].mean()
    return cvar_value