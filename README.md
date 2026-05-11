# Equity Alpha Engine (Cross-Sectional Quant Framework)
> **Language**: [English] | [한국어](./README_kr.md)
## Overview
This project implements a **cross-sectional equity trading framework** focused on the full pipeline:

> **Signal → Portfolio Construction → Execution → Evaluation**

The goal is to study how alpha signals behave **after realistic portfolio constraints**, including turnover and transaction costs.

---

## Key Components

### 1. Data Pipeline
- Supports Korean and US equities
- Universe selection via **liquidity filter (top-N by dollar volume)**
- Handles missing data with masking and controlled forward-fill
- Outputs aligned **price, volume, and return matrices**

---

### 2. Alpha Signals
Implemented cross-sectional signals:

- Momentum (medium-term)
- Volatility-adjusted momentum
- Mean reversion (short-term)
- Residual momentum (market-neutralized)

All signals are evaluated using **Information Coefficient (IC)** across multiple horizons.

---

### 3. Portfolio Construction

Two portfolio construction approaches are implemented:

#### (1) Fast Heuristic Optimizer
- Volatility scaling
- cross-sectional normalization + direct signal-to-weight mapping (non-optimization)
- Turnover-aware adjustment (soft constraint)
- Designed for **speed and robustness**

#### (2) Factor Projection Optimizer (Long-Short)

- Partial market neutralization via factor residualization
- Residual-based long/short construction (not fully beta-neutral constrained)
- No covariance estimation required
- Residual-based alpha extraction
- Rank-driven portfolio formation
- Robust to market-wide shocks

#### (3) Convex Optimization (Mean-Variance)
- Currently not used (for future research)
- Objective: maximize αᵀw − λ wᵀΣw
- Rolling covariance estimation (lookahead-free)
- Uses precomputed covariance matrix (external estimation required)
- Includes:
  - Market-neutral constraint
  - Gross leverage constraint
  - Position limits
  - Turnover constraint

---

### 4. Execution & Cost Modeling
- Periodic rebalancing with **portfolio drift**
- Explicit turnover tracking
- Transaction cost deduction from PnL

---

### 5. Evaluation Metrics
- CAGR
- Sharpe Ratio
- Maximum Drawdown (MDD)
- Information Coefficient (IC)
- Transfer Coefficient (TC)

> - IC measures predictive power of signals, TC measures correlation between signal and implemented portfolio weights (signal realization efficiency)

---

### 6. Strategy Selection & Ensemble
- IC-based filtering (StrategyPruner)
- Correlation-based pruning (StrategyPruner)
- Softmax-weighted ensemble of strategy PnL (SoftmaxEnsembleEngine)

---

## Key Findings

- **IC does not directly translate to PnL**  
  Portfolio construction can significantly distort signal effectiveness.

- **Execution plays a critical role**  
  Turnover and transaction costs materially impact realized performance.

- **Optimization trade-offs exist**  
  Mean-variance optimization can dilute weak signals, while heuristic methods often preserve alpha more effectively.

- **Alpha dilution is a key issue**  
  Rank-based filtering improved signal-to-noise ratio.

---

## Project Structure

```project/
├── datahandler/ # Data loading and preprocessing
├── strategy/ # Signal generation and backtesting engine
├── optimizer/ # Portfolio construction (heuristic / CVX)
├── pruner/ # Signal filtering (IC, correlation)
├── softmax/ # Strategy ensemble
├── main.py # Entry point
└── README.md
```

## Usage

```python
data = Datahandler.DataHandler_KR(start="2016-06-01", end="2026-03-31", universe="KOSPI")

price = data.get_price()
returns = data.get_returns()
volume = data.get_volume()

engine = Strategy.StrategyEngine(price, returns, volume)

engine.add_signal("momentum", lambda p: p.shift(20).pct_change(60))

engine.compute_signals()

results = engine.run_all()
pruner = Pruner.StrategyPruner(results, ic_threshold=0.3, corr_threshold=0.7)

filtered_results = pruner.run()

soft_engine = Softmax.SoftmaxEnsembleEngine(filtered_results)

output = soft_engine.run()