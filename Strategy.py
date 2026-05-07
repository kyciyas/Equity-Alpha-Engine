import pandas as pd
import numpy as np
import Optimizer

class StrategyEngine:
    def __init__(self, price, returns, volume, cost_rate=0.002, ic_window=60):
        self.price = price
        self.returns = returns
        self.volume = volume

        self.cost_rate = cost_rate
        self.ic_window = ic_window

        self.signals = {}
        self.signal_values = {}

        self.ic_scores = {}
        self.strategy_weights = None
        self.days_per_year = 252
        self.drift_days = 5

        self.universe = sorted(price.columns)
    # SIGNAL REGISTRATION
    def add_signal(self, name, func, neutralize=False):
        self.signals[name] = {"func": func, "neutralize": neutralize}

    def compute_ic(self, signal, horizons=[1, 5, 10, 20]):

        price = self.price
        ic_result = {}

        for h in horizons:
            future_ret = price.shift(-h) / price - 1

            ic_list = []

            for date in signal.index:
                s = signal.loc[date]
                r = future_ret.loc[date]

                df = pd.concat([s, r], axis=1).dropna()

                if len(df) < 20:
                    continue

                ic = df.iloc[:, 0].corr(df.iloc[:, 1])
                if np.isfinite(ic):
                    ic_list.append(ic)

            ic_result[h] = round(float(np.mean(ic_list)), 3) if len(ic_list) > 0 else 0.0

        return ic_result

    def compute_signals(self):
        for name, cfg in self.signals.items():
            sig = cfg["func"](self.price)

            sig = sig.reindex_like(self.price)

            # optional neutralization hook
            if cfg["neutralize"]:
                sig = sig.sub(sig.mean(axis=1), axis=0)

            self.signal_values[name] = sig


    # PORTFOLIO CONSTRUCTION (FIXED)
    def run(self, signal_name, mode="fast"):

        signal = self.signal_values[signal_name]

        # mode:
        #    "fast"         heuristic
        #    "projection"   factor projection

        if mode == "fast":
            optimizer = Optimizer.FastPortfolioOptimizer()
        elif mode == "projection":
            optimizer = Optimizer.FactorProjectionOptimizer()
        else:
            raise ValueError("mode must be 'fast' or 'projection'")

        weights = pd.DataFrame(0.0, index=signal.index, columns=self.universe)

        prev_w = None

        for i, date in enumerate(signal.index):
            # 1. Rebalanced skip (drift)
            if i % self.drift_days != 0:
                if prev_w is not None:
                    ret = self.returns.loc[date].reindex(prev_w.index).fillna(0.0)

                    prev_w = prev_w * (1 + ret)
                    prev_w = prev_w / (prev_w.abs().sum() + 1e-8)

                    weights.loc[date, prev_w.index] = prev_w
                continue

            # 2. Signal process
            processed_signal = signal.loc[date].reindex(self.universe).fillna(0)
            alpha = processed_signal.copy()

            # 3. prev_w alignment
            if prev_w is None:
                prev_w = pd.Series(0.0, index=self.universe)
            else:
                prev_w = prev_w.reindex(self.universe).fillna(0.0)

            # 4. returns window
            returns_window = self.returns.loc[:date, self.universe]
            if mode == "projection":
                windows = returns_window.tail(60)
            elif mode == 'fast':
                windows = returns_window.tail(20)

            # 5. optimizer
            weight_opt = optimizer.solve(alpha, windows, prev_w)

            # 6. inertia
            if prev_w is not None:
                weight_temp = 0.7 * prev_w + 0.3 * weight_opt
            else:
                weight_temp = weight_opt

            # scale control (NOT hard L1 normalization)
            scale = np.linalg.norm(weight_temp)

            if scale > 0:
                weight_temp = weight_temp / scale

            prev_w = weight_temp.copy()
            weights.loc[date, weight_temp.index] = weight_temp

        # 8. PnL + cost
        weights = weights.shift(1)
        gross_pnl = (weights * self.returns).sum(axis=1)

        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover * self.cost_rate

        pnl = gross_pnl - cost
        return pnl, weights

    # RUN ALL
    def run_all(self, mode="fast"):
        results = {}

        for name in self.signals.keys():
            pnl, weights = self.run(name, mode=mode)

            stats = self.evaluate(pnl)
            ic = self.compute_ic(self.signal_values[name])

            results[name] = {
                "pnl": pnl,
                "weights": weights,
                "stats": stats,
                "ic": ic
            }

        return results


    # STATS
    def evaluate(self, pnl):

        pnl = pnl.dropna()
        cum = (1 + pnl).cumprod()

        cagr = cum.iloc[-1] ** (self.days_per_year / len(cum)) - 1
        sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(self.days_per_year)
        mdd = (cum / cum.cummax() - 1).min()

        return {
            "CAGR": round(float(cagr), 3),
            "Sharpe": round(float(sharpe), 3),
            "MDD": round(float(mdd), 3)
        }