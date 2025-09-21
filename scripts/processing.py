import numpy as np
import pandas as pd
    


def get_returns(Prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute discrete returns from price data.
    
    Parameters :
        Prices : pd.DataFrame : Price time series.
    
    Returns:
        pd.DataFrame
    """
    return (Prices / Prices.shift(1) - 1).dropna()


def get_logreturns(Prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute discrete log-returns from price data.
    
    Parameters :
        Prices : pd.DataFrame : Price time series.
    
    Returns:
        pd.DataFrame
    """
    
    return (np.log(Prices / Prices.shift(1))).dropna()



