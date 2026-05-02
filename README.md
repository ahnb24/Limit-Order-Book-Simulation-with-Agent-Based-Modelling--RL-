# 📊 Limit Order Book Simulation with Agent-Based Modelling

## 📌 Overview

This project implements an agent-based simulation of a financial market using a continuous double-auction Limit Order Book (LOB).

The objective is to study how interactions between:
- Zero-Intelligence Traders (ZIT)
- Reinforcement Learning (RL) Traders

affect market dynamics such as volatility, price formation, and profitability.

---

## ⚙️ Model Description

### 🏦 Limit Order Book (LOB)

The market operates as a continuous double auction:

- Buy order executes if:
  P_buy ≥ best ask

- Sell order executes if:
  P_sell ≤ best bid

- Mid-price:
  M_t = (best bid + best ask) / 2

Orders follow price-time priority.

---

## 🤖 Agent Types

### 🔹 Zero-Intelligence Traders (ZIT)

- Submit random limit orders
- Provide liquidity around a fundamental price

### 🔹 Reinforcement Learning Traders (RL)

- Use tabular Q-learning
- Learn optimal trading strategies based on PnL

Reward function:
R_t = ΔPnL - λ_inv * |inventory| - λ_hold * 1{hold}

---

## 📊 Outputs & Metrics

The simulation tracks:

- Mid-price
- Excess kurtosis of returns
- ACF of squared returns (volatility clustering)
- Profit & Loss (PnL)
- Inventory dynamics
- RL action frequencies

---

## 📚 Experiments & Results

### 1. Market Composition (ω)

We vary the fraction of ZIT traders (ω):

| ω | Excess Kurtosis | ACF Lag 1 | ACF Lag 2 | ACF Lag 3 | ACF Lag 4 |
|--|--|--|--|--|--|
| 0.1 | 22.90 | 0.1021 | 0.0860 | 0.0028 | -0.0006 |
| 0.5 | 12.56 | 0.2292 | 0.1180 | 0.1168 | 0.1442 |
| 0.9 | 1.60  | 0.1760 | 0.0646 | 0.0322 | -0.0225 |

**Insights:**
- Low ω (more RL) → extreme fat tails and instability  
- Medium ω → strongest volatility clustering  
- High ω (more ZIT) → stable, near-normal behavior  

---

### 2. Parameter Sensitivity

#### Inventory Penalty (λ_inv)

- λ_inv = 0 → RL agents take large directional positions (speculative behavior)
- λ_inv = 5 → RL agents keep inventory close to zero (market-making behavior)

---

#### Holding Penalty (λ_hold)

- λ_hold = 0 → agents are passive, lower trading activity  
- λ_hold = 5 → higher trading frequency and increased volatility  

---

#### Exploration Rate (ε₀)

- ε₀ = 0.1 → stable learning, smoother PnL  
- ε₀ = 0.9 → noisy learning, unstable and lower profits  

---

### 3. Stylised Facts Calibration

| Metric | Baseline | Calibrated |
|--|--|--|
| Excess Kurtosis | 12.56 | 3.20 |
| ACF Lag 1 | 0.2292 | 0.2972 |
| ACF Lag 2 | 0.1180 | 0.1614 |
| ACF Lag 3 | 0.1168 | 0.1710 |
| ACF Lag 4 | 0.1442 | 0.0835 |

**Result:**  
The calibrated model successfully reproduces:
- Fat-tailed returns  
- Volatility clustering  

---

### 4. RL Model Extension

Modified reward:

Original:
λ_inv * |inventory|

Modified:
λ_inv * inventory²

| Metric | Baseline | Modified RL |
|--|--|--|
| Excess Kurtosis | 12.56 | 16.65 |
| ACF Lag 1 | 0.2292 | 0.0734 |
| ACF Lag 2 | 0.1180 | 0.0720 |
| ACF Lag 3 | 0.1168 | 0.0306 |
| ACF Lag 4 | 0.1442 | 0.1023 |

**Insight:**
- Higher kurtosis → more extreme price movements  
- Lower ACF → reduced volatility persistence  

---

## 📂 Repository Structure
