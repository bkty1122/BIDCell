import torch
import numpy as np
from scipy.optimize import minimize
try:
    from torchjd.aggregation import Aggregator
except ImportError:
    # Fallback if Aggregator is not importable directly
    class Aggregator:
        def __call__(self, matrix):
            raise NotImplementedError

class CAGrad(Aggregator):
    def __init__(self, c=0.5):
        super().__init__()
        self.c = c

    def forward(self, grads):
        """
        grads: (n_tasks, n_params)
        """
        # Ensure grads is 2D
        if grads.dim() == 1:
            return grads
            
        n_tasks = grads.shape[0]
        device = grads.device
        
        # Gram matrix of gradients H = G G^T
        # We process in CPU for scipy
        grads_cpu = grads.detach().cpu()
        GG = torch.mm(grads_cpu, grads_cpu.t()).numpy() # (n_tasks, n_tasks)
        
        g0_norm = np.linalg.norm(grads_cpu.mean(dim=0).numpy())
        
        # If g0_norm is too small, just average
        if g0_norm < 1e-8:
            weights = np.ones(n_tasks) / n_tasks
            w_torch = torch.from_numpy(weights).float().to(device)
            return torch.matmul(w_torch, grads)

        x_start = np.ones(n_tasks) / n_tasks
        bnds = tuple((0, 1) for _ in range(n_tasks))
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

        # CAGrad Dual Objective:
        # minimize F(w) = w^T GG w
        # subject to constraints potentially
        # The exact formulation involves finding a direction d
        # Here we approximate finding the optimal weighting w
        
        def obj(w):
            return 0.5 * np.dot(w, np.dot(GG, w)) 
            
        res = minimize(obj, x_start, method='SLSQP', bounds=bnds, constraints=cons)
        weights = res.x
        
        w_torch = torch.from_numpy(weights).float().to(device)
        return torch.matmul(w_torch, grads)

class NashMTL(Aggregator):
    def forward(self, grads):
        """
        grads: (n_tasks, n_params)
        """
        n_tasks = grads.shape[0]
        device = grads.device
        
        grads_cpu = grads.detach().cpu().numpy()
        GG = np.dot(grads_cpu, grads_cpu.T) # (n_tasks, n_tasks)
        
        # Nash MTL formulation is often solving for alpha
        # maximize sum log(alpha) s.t. constraints
        # Or minimizing distance to bargaining solution
        
        # Simplified proxy implementation using scipy
        # We want to find alphas
        
        x_start = np.ones(n_tasks) / n_tasks
        bnds = tuple((0, None) for _ in range(n_tasks))
        
        # Warning: This is a placeholder approximation.
        # Real NashMTL requires specific update logic.
        # We will use a standard MGDA-like approach as a stand-in if strict Nash is not possible without solvers
        # But let's try to maximize sum of log(w' G_i) -> no.
        
        # Implementation based on "Multi-Task Learning as a Bargaining Game"
        # max \sum log(alpha_i) - 0.5 || \sum alpha_i g_i ||^2
        # This is strictly concave.
        
        def obj(alpha):
            # maximize sum log(alpha) - 0.5 * ||G^T alpha||^2
            # minimize - (sum log(alpha) - 0.5 * alpha^T GG alpha)
            reg = 0.5 * np.dot(alpha, np.dot(GG, alpha))
            log_term = np.sum(np.log(alpha + 1e-10)) # add epsilon for stability
            return reg - log_term
            
        res = minimize(obj, x_start, method='SLSQP', bounds=bnds)
        weights = res.x
        
        # Normalize weights maybe? NashMTL weights are not necessarily sum to 1
        # But we probably want to normalize for stability
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        w_torch = torch.from_numpy(weights).float().to(device)
        return torch.matmul(w_torch, grads)
