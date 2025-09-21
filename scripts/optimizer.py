import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict

class Optimizer_Dorado:
    """
    This class implement the Dorado's technique to linearize the Max - Drawdown problem :
    maximize v, with respect to r_t @ w >= v for all t in window.
    
    """
    
    def __init__(self, returns_window : pd.DataFrame, min_expected_return : Optional[float] = None, max_weight : str = 0.5, long_only : bool = True):
        
        self.returns = returns_window.copy()
        self.max_weight = max_weight
        self.columns = self.returns.columns
        self.n = len(self.columns)
        self.min_expected_return = min_expected_return # This is daily
        self.long_only = long_only
        
        
    def solve(self, solver: Optional[str] = None, verbose : bool = False, alpha: float = 0.6) -> Dict[str, np.ndarray]:

        """
        Solve to get optimal weights
        
        Parameters : 
            solver : Optional[str] : the solver use to access the problem
            verbose : bool : defaults to False, True for debugging
            alpha : float : Balances risk 
            
        Returns:
            Dict[str, np.ndarray] : it stores the optimal weights, min expected return and the status 
        
        """
        
        R = self.returns.values
        w = cp.Variable(self.n)
        v = cp.Variable()
        
        mu = self.returns.mean(axis=0).values
        
        
        objective = cp.Maximize(alpha * v + (1 - alpha) * mu @ w)
        
        constraints = []
        
        constraints.append(cp.sum(w) == 1)
        if self.long_only:
            constraints.append(w >= 0)
            
        constraints.append(w <= self.max_weight)
        
        
        # Constraint on the long run
        
        if self.min_expected_return:
            mu = self.returns.mean(axis = 0).values
            constraints.append(mu @ w >= self.min_expected_return)
            
        # Epigraph constraint
        
        constraints.append(R @ w >= v)
            
        prob = cp.Problem(objective, constraints)
        
        if solver is None:
            prob.solve(solver=cp.OSQP, verbose=verbose)  
        else:
            prob.solve(solver=solver, verbose=verbose)

        return {
            'weights': w.value if w.value is not None else np.full(self.n, np.nan),
            'v': float(v.value) if v.value is not None else float('nan'),
            'status': prob.status
        }
            
        
        
        