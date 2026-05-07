import pandas as pd
import cvxpy as cp
import numpy as np

class FastPortfolioOptimizer:
    def __init__(self, max_leverage=1.0, vol_window=20, turnover_penalty=0.2):
        self.max_leverage = max_leverage
        self.vol_window = vol_window
        self.turnover_penalty = turnover_penalty  # 이제 "mixing weight"

    def solve(self, alpha, returns, prev_w=None):

        # 1. volatility scaling
        vol = returns.std(axis=0)
        vol = vol.reindex(alpha.index).replace(0, 1e-6)
        alpha_adj = alpha / vol

        # 2. centering (market neutral)
        alpha_adj = alpha_adj - alpha_adj.mean()

        # 3. raw weights
        if alpha_adj.abs().sum() == 0:
            return pd.Series(0.0, index=alpha.index)

        w = alpha_adj / (alpha_adj.abs().sum() + 1e-8)

        # 4. soft turnover control
        if prev_w is not None:
            prev_w = prev_w.reindex(w.index).fillna(0.0)
            delta = w - prev_w

            # turnover budget
            budget = self.turnover_penalty

            total_turnover = np.abs(delta).sum()

            if total_turnover > budget:
                scale = budget / (total_turnover + 1e-8)
                delta = delta * scale

            w = prev_w + delta

        # 5. leverage normalize
        w = w * self.max_leverage / (np.abs(w).sum() + 1e-8)

        return pd.Series(w, index=alpha.index)

class FactorProjectionOptimizer:

    def __init__(self, n_long=50, n_short=50):
        self.n_long = n_long
        self.n_short = n_short

    def neutralize(self, alpha, returns_window):

        # 1. clean data
        returns_window = returns_window.dropna(axis=1, thresh=int(len(returns_window) * 0.8))
        returns_window = returns_window.dropna(axis=0)

        if returns_window.shape[0] < 10 or returns_window.shape[1] < 10:
            return alpha

        # 2. market factor
        market = returns_window.mean(axis=1)

        if market.std() < 1e-6:
            return alpha

        # 3. build matrices
        F = market.values.reshape(-1, 1)  # T x 1
        R = returns_window.values  # T x N

        try:
            beta = np.linalg.lstsq(F, R, rcond=None)[0]

            factor_part = F @ beta
            residual = R - factor_part

            alpha_clean = pd.Series(residual[-1], index=returns_window.columns)

            return alpha_clean.reindex(alpha.index).fillna(0)

        except:
            return alpha

    def solve(self, alpha, returns_window, prev_w=None):
        # 1. factor neutralization
        alpha_clean = self.neutralize(alpha, returns_window)

        # 2. rank-based selection
        alpha_clean = alpha_clean.dropna()

        if len(alpha_clean) < (self.n_long + self.n_short):
            return pd.Series(0.0, index=alpha.index)

        long = alpha_clean.nlargest(self.n_long)
        short = alpha_clean.nsmallest(self.n_short)

        w = pd.Series(0.0, index=alpha_clean.index)
        w[long.index] = 1.0
        w[short.index] = -1.0

        # 3. normalize
        w = w / (w.abs().sum() + 1e-8)

        return w.reindex(alpha.index).fillna(0.0)

# currently not used
class PortfolioOptimizer:

    def __init__(self, risk_aversion=1.0, turnover_penalty=0.1, max_weight=0.05):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.max_weight = max_weight

    def solve(self, alpha, cov, prev_w=None):
        index = alpha.index

        alpha = alpha.values
        cov = cov.values

        n = len(alpha)
        w = cp.Variable(n)

        objective = cp.Maximize(alpha @ w - self.risk_aversion * cp.quad_form(w, cov))

        constraints = [cp.sum(w) == 0, cp.norm1(w) <= 1, w <= self.max_weight, w >= -self.max_weight]

        if prev_w is not None:
            prev_arr = prev_w.values
            constraints.append(cp.norm1(w - prev_w.values) <= 0.2)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if w.value is None:
            return pd.Series(np.zeros(n), index=index)

        return pd.Series(np.array(w.value).flatten(), index=index)