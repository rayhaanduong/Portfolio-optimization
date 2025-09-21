from optimizer import Optimizer_Dorado
from risk import sharpe_ratio, max_drawdown, volatility, sortino_ratio, cvar

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import statsmodels.api as sm

def get_rebalance_dates(dates : pd.DatetimeIndex, roll : str = "month_start") ->List[pd.Timestamp]:
    
    """
    Return a list of rebalancing dates, defaults to monthly rebalancing
    
    Parameters :
        dates : pd.DatetimeIndex
        roll : str : defaults to month_start -> we rablance the first day of the month, otherwise the last day
        
    """
    
    df = pd.Series(index = dates, data = dates.month)
    
    if roll == "month_start":    
        
        return df.groupby(df.index.to_period('M')).apply(lambda x: x.index.min()).tolist()
    
    else:
        
        return df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max()).tolist()
        
        

class PortfolioManager:
    
    def __init__(self, prices : pd.DataFrame):
        
        self.prices = prices.sort_index()
        self.returns = prices.pct_change().dropna()
        self.tickers = list(self.prices.columns)
        
    
           
    def run_monthly_rebalance(self, start_date : str, end_date : str, trading_days : int = 252, 
                              min_expected_return : Optional[float] = None, max_weight : float = 0.5, 
                              long_only : bool = True, transaction_cost_bps : float = 0.0, solver : Optional[str] = None, 
                              verbose : bool = False, hedge_against: Optional[pd.Series] = None, vix : Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        
        """
        Run Monthly rebalancing using the Dorado Optimizer
        
        Parameters : 
            start_date : str (YYYY-MM-DD)
            end_date : str (YYYY-MM-DD)
            trading_days : int : defaults to 252, change to 365 if crypto assets
            min_expected_return : Optional[float] : The minimum daily returns we expect
            max_weight : float : defaults to 0.5 (to avoid portofolio of 1 stock only)
            long_only : bool : defaults to True 
            transaction_cost_bps : float : defaults to 0.0 for debugging, could defaults to 5 bps per round trip
            solver : Optional[str] : We can choose our solver
            verbose : bool : defaults to False
            hedge_against : Optional[pd.Series] : for example Nasdaq
            vix : Optional[pd.Series] : indicator of volatility
            
        Returns :
            Dict[str, pd.DataFrame] : Returns dict with weights_over_time (DataFrame), daily_portf_returns (Series), metrics
        
        """
        
        all_dates = self.returns.loc[start_date:end_date].index
        rebal_dates = get_rebalance_dates(all_dates, roll = "month_start")
        rebal_dates = [d for d in rebal_dates if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)]   
        
        weights_history = []
        dates_history = []
        daily_portf_ret = pd.Series(index=all_dates, dtype=float)
        prev_weights = None     
        
            
            
        for j, reb_date in enumerate(rebal_dates):
        
            window_end = reb_date - pd.Timedelta(days = 1)
            window_start = window_end - pd.Timedelta(days = trading_days) # Maybe it's a parameter that we will change 
            window_returns = self.returns.loc[window_start:window_end, self.tickers].dropna(axis = 1, how = 'all')
            
            valid_assets = window_returns.columns.tolist()
            
            
            if len(valid_assets) == 0.3:
                
            # We use last weights or 1 / n
            
                weights = prev_weights if prev_weights is not None else np.ones(len(self.tickers)) / len(self.tickers)
                weights_history.append(weights)
                dates_history.append(reb_date)
                continue
            
            
            if vix is not None :
            
                vix_now = vix.loc[reb_date]
                
                if vix_now > 40:       
                    alpha_local = 0.8
                elif vix_now > 15:    
                    alpha_local = 0.5
                else:                
                    alpha_local = 0.3
                    
            else :
                alpha_local = 1
                
            
            
            
            optimizer = Optimizer_Dorado(window_returns[valid_assets], min_expected_return = min_expected_return, max_weight = max_weight, long_only = long_only)
            
            sol = optimizer.solve(solver, verbose, alpha_local)
            w_vec = sol["weights"]    
            
            mapped = np.zeros(len(self.tickers))
            index_map = {t: j for j, t in enumerate(valid_assets)}
            
            for i, t in enumerate(self.tickers):
                if t in index_map:
                    mapped[i] = w_vec[index_map[t]]
                    
                    
            # If no weights we do 1 / n
                    
            if mapped.sum() > 0:
                mapped /= mapped.sum()
            else:
                mapped = np.ones(len(self.tickers)) / len(self.tickers)
            

            # Forward period :
            
            if j + 1 < len(rebal_dates):
                
                period_idx = self.returns.loc[reb_date : rebal_dates[j+1]].index
            
            else:
                
                period_idx = self.returns.loc[reb_date:end_date].index
                
            forward_returns = self.returns.loc[period_idx, self.tickers].fillna(0.0).dot(mapped)
            
            
            # Hedge if not None
            
            if hedge_against is not None:
                
                beta_window = hedge_against.loc[window_start:window_end].reindex(window_returns.index).fillna(0.0)
                X = sm.add_constant(beta_window)
                model = sm.OLS(window_returns.dot(mapped), X).fit()
                beta = model.params.iloc[1]
                
                forward_returns -=  beta * hedge_against.loc[period_idx]
                
            
            # Transaction costs
         
            if prev_weights is not None and transaction_cost_bps > 0:
                turnover = np.abs(mapped - prev_weights).sum()
                cost = turnover * (transaction_cost_bps / 10000)  # convert bps to fraction
                forward_returns.iloc[0] -= cost
                            
            prev_weights = mapped
            weights_history.append(mapped)
            dates_history.append(reb_date)
            
            daily_portf_ret.loc[period_idx] = forward_returns
            

        # Construct weights DataFrame
        weights_df = pd.DataFrame(weights_history, index=dates_history, columns=self.tickers)
        
        # Construct the portolios
        daily_portf_ret = daily_portf_ret.dropna()
        cum_ret = (1 + daily_portf_ret).cumprod()
        
        # Metrics
        md = max_drawdown(cum_ret)
        sharpe = sharpe_ratio(daily_portf_ret, periods_per_year = 252)
        v = volatility(daily_portf_ret, periods_per_year = 252)
        sortino = sortino_ratio(daily_portf_ret, periods_per_year = 252)
        cvar_ = cvar(daily_portf_ret)
        
        return {
            "weights": weights_df,
            "daily_returns": daily_portf_ret,
            "cumulative_returns": cum_ret,
            "max_drawdown": md,
            "annualized_sharpe": sharpe,
            "volatility" : v,
            "sortino" : sortino,
            "cvar" : cvar_
        }
    
        
    
        
        
        
        
        
        
    def run_monthly_1overN(self, start_date : str, end_date : str, transaction_cost_bps : float, hedge_against : Optional[str] = None,  trading_days : int = 252) -> Dict[str, pd.DataFrame]:
        
        """
        Run Monthly rebalancing using a simple 1/N strategy
        
        Parameters : 
            start_date : str (YYYY-MM-DD)
            end_date : str (YYYY-MM-DD)
            transaction_cost_bps : float : defaults to 0.0 for debugging, could defaults to 5 bps per round trip
            hedge_against : Optional[str] : for example against Nasdaq
            trading_days : int : defaults to 252
        
            
        Returns :
            Dict[str, pd.DataFrame] : Returns dict with weights_over_time (DataFrame), daily_portf_returns (Series), metrics
        
        """
        
        all_dates = self.returns.loc[start_date:end_date].index
        rebal_dates = get_rebalance_dates(all_dates, roll="month_start")
        rebal_dates = [d for d in rebal_dates if d >= pd.to_datetime(start_date) and d <= pd.to_datetime(end_date)]   

        weights_history = []
        dates_history = []

        n_assets = len(self.tickers)
        equal_weight = np.ones(n_assets) / n_assets  # 1/N weights
        
        daily_portf_ret = pd.Series(index=all_dates, dtype=float)
        

        for i, reb_date in enumerate(rebal_dates):
           
            weights_history.append(equal_weight)
            dates_history.append(reb_date)

            if i + 1 < len(rebal_dates):
                
                period_idx = self.returns.loc[reb_date : rebal_dates[i+1]].index
                
            else:
                
                period_idx = self.returns.loc[reb_date:end_date].index
                
                
            forward_returns = self.returns.loc[period_idx, self.tickers].fillna(0.0).dot(equal_weight)
            
            if hedge_against is not None:
                
                window_end = reb_date - pd.Timedelta(days=1)
                window_start = window_end - pd.Timedelta(days=trading_days)
                
                lookback_portfolio = self.returns.loc[window_start:window_end, self.tickers].fillna(0.0).dot(equal_weight)
                lookback_benchmark = hedge_against.loc[window_start:window_end].reindex(lookback_portfolio.index).fillna(0.0)
                 
                X = sm.add_constant(lookback_benchmark)
                model = sm.OLS(lookback_portfolio, X).fit()
                beta = model.params.iloc[1]
                
                forward_returns -=  beta * hedge_against.loc[period_idx]
                
            
            daily_portf_ret.loc[period_idx] = forward_returns
            
            
        # Construct weights DataFrame
        weights_df = pd.DataFrame(weights_history, index=dates_history, columns=self.tickers)
        
        # Construct the portolios
        daily_portf_ret = daily_portf_ret.fillna(0.0)
        cum_ret = (1 + daily_portf_ret).cumprod()
        
        # Metrics
        md = max_drawdown(cum_ret)
        sharpe = sharpe_ratio(daily_portf_ret, periods_per_year = 252)
        v = volatility(daily_portf_ret, periods_per_year = 252)
        
        
        return {
            "weights": weights_df,
            "daily_returns": daily_portf_ret,
            "cumulative_returns": cum_ret,
            "max_drawdown": md,
            "annualized_sharpe": sharpe,
            "volatility" : v

        }
    
        
                































            
        
            
            
             