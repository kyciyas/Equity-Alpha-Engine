import numpy as np
import pandas as pd

class StrategyPruner:

    def __init__(self, results, ic_threshold=0.0, corr_threshold=0.7):
        """
        results:
            {
                strategy_name: {
                    'pnl': pd.Series,
                    'stats': dict,
                    'ic': dict
                }
            }
        """
        self.results = results
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold

    def filter_by_ic(self):

        filtered = {}

        for name, res in self.results.items():
            ic_dict = res['ic']
            vals = list(ic_dict.values())
            mean_ic = np.mean(vals) if len(vals) > 0 else 0.0

            if mean_ic > self.ic_threshold:
                filtered[name] = res

        return filtered

    def compute_corr_matrix(self, results):

        names = list(results.keys())
        pnl_matrix = pd.DataFrame(
            {n: results[n]['pnl'] for n in names}
        )

        corr = pnl_matrix.corr()

        return corr

    def filter_by_correlation(self, results):

        names = list(results.keys())
        corr = self.compute_corr_matrix(results)

        keep = []

        for name in names:

            if len(keep) == 0:
                keep.append(name)
                continue

            too_similar = False

            for k in keep:
                if abs(corr.loc[name, k]) > self.corr_threshold:
                    too_similar = True
                    break

            if not too_similar:
                keep.append(name)

        return {k: results[k] for k in keep}

    def run(self):

        # 1. IC filter
        ic_filtered = self.filter_by_ic()

        # 2. correlation filter
        final = self.filter_by_correlation(ic_filtered)

        return final