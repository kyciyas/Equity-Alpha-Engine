import Datahandler
import Strategy
import Pruner
import Softmax

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def compute_transfer_coefficient(signal, weights):
    tc_list = []

    for date in signal.index:
        s = signal.loc[date]
        w = weights.loc[date]

        df = pd.concat([s, w], axis=1).dropna()

        if len(df) < 20:
            continue

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        if x.std() < 1e-8 or y.std() < 1e-8:
            continue

        tc = x.corr(y)
        if np.isfinite(tc):
            tc_list.append(tc)

    return np.mean(tc_list)

data = Datahandler.DataHandler_KR(start="2016-06-01", end="2024-12-31", universe="KOSPI")
# data = Datahandler.DataHandler_US(start="2016-06-01", end="2026-3-31")

price = data.get_price()
returns = data.get_returns()
volume = data.get_volume()
mask = data.get_mask()

engine = Strategy.StrategyEngine(price, returns, volume)

engine.add_signal("momentum", lambda p: p.shift(5).pct_change(60, fill_method=None))
engine.add_signal("mean_rev", lambda p: -(p.shift(1).pct_change(5, fill_method=None).rolling(3).mean()))

engine.compute_signals()

results = engine.run_all(mode = 'fast')
for name, res in results.items():
    print(name, res['stats'], res['ic'])
    tc = compute_transfer_coefficient(
        engine.signal_values[name],
        res['weights']
    )
    print(name, "TC:", round(tc, 3))

pruner = Pruner.StrategyPruner(results, ic_threshold=0.001, corr_threshold=0.5)
filtered_results = pruner.run()

for name, res in filtered_results.items():
    print(name)

soft_engine = Softmax.SoftmaxEnsembleEngine(filtered_results)

output = soft_engine.run()

print(round(output["pnl"].cumsum().iloc[-1], 3))

for name, res in filtered_results.items():
    res['pnl'].cumsum().plot(label=name)

plt.legend()
plt.show()