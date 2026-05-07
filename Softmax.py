import numpy as np

class SoftmaxEnsembleEngine:
    def __init__(self, results, cost_rate=0.002):
        """
        results:
            {
                'strategy_name': {
                    'pnl': pd.Series,
                    'stats': dict,
                    'ic': dict
                }
            }
        """
        self.results = results
        self.cost_rate = cost_rate

        self.weights_history = None
        self.scores = None

    def score_strategy(self, stats, ic):

        sharpe = stats['Sharpe']
        cagr = stats['CAGR']
        mdd = abs(stats['MDD'])
        ic_mean = np.mean(list(ic.values()))

        score = (sharpe * 0.5 + cagr * 0.3 + ic_mean * 0.3 - mdd * 0.2)
        return score

    def softmax(self, x):

        x = np.array(x)
        x = x / (np.std(x) + 1e-8)   # stability boost

        e = np.exp(x - np.max(x))
        return e / e.sum()

    def compute_weights(self):

        names = []
        scores = []

        for name, res in self.results.items():
            s = self.score_strategy(res['stats'], res['ic'])
            names.append(name)
            scores.append(s)

        weights = self.softmax(scores)

        self.scores = dict(zip(names, scores))
        self.weights = dict(zip(names, weights))

        return self.weights

    def build_portfolio(self):

        if not hasattr(self, "weights"):
            self.compute_weights()

        combined_pnl = None

        for name, w in self.weights.items():
            pnl = self.results[name]['pnl']

            if combined_pnl is None:
                combined_pnl = w * pnl
            else:
                combined_pnl += w * pnl

        return combined_pnl

    def run(self):

        weights = self.compute_weights()
        pnl = self.build_portfolio()

        return {
            "pnl": pnl,
            "weights": weights,
            "scores": self.scores
        }