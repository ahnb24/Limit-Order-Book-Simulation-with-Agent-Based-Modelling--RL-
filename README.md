# 📊 Limit Order Book Simulation with Agent-Based Modelling

## 📌 Overview

This project implements an agent-based simulation of a financial market using a continuous double-auction Limit Order Book (LOB).

The objective is to study how interactions between:
- Zero-Intelligence Traders (ZIT)
- Reinforcement Learning (RL) Traders

affect market dynamics such as volatility, price formation, and profitability.

The model reproduces key stylised facts observed in real financial markets, including fat-tailed returns and volatility clustering.

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
- Prices are drawn around a fundamental value
- Provide liquidity to the market

### 🔹 Reinforcement Learning Traders (RL)

- Use tabular Q-learning
- Observe best bid and best ask
- Choose from:
  - Market buy
  - Market sell
  - Limit buy
  - Limit sell
  - Hold

- Reward function:
  R_t = ΔPnL - λ_inv * |inventory| - λ_hold * 1{hold}

---

## 🔁 Simulation Process

At each time step:

1. Select an agent
2. Observe market state
3. Take action:
   - ZIT → random limit order
   - RL → ε-greedy policy
4. Execute trades
5. Update inventory and cash
6. Update Q-values (RL only)
7. Remove expired orders

---

## 📊 Outputs & Metrics

The simulation tracks:

- Mid-price evolution
- Return distribution (kurtosis)
- Volatility clustering (ACF of squared returns)
- Profit & Loss (PnL)
- Inventory dynamics
- RL action frequencies

---

## 📚 Experiments

### 1. Market Composition (ω)

The fraction of ZIT traders (ω) is varied:

- ω = 0.1 → mostly RL traders → high volatility and fat tails  
- ω = 0.5 → mixed market → strongest volatility clustering  
- ω = 0.9 → mostly ZIT traders → stable and near-normal behavior  

---

### 2. Parameter Sensitivity

We analyze the effect of:

- Inventory penalty (λ_inv)
- Holding penalty (λ_hold)
- Exploration rate (ε₀)

Key insights:

- Higher λ_inv → more risk-averse behavior  
- Higher λ_hold → more trading activity  
- Higher ε₀ → unstable learning and lower profitability  

---

### 3. Stylised Facts Calibration

We tune parameters to reproduce:

- Fat tails in returns
- Volatility clustering

Achieved by combining:
- High ZIT participation
- Longer order persistence
- Flexible RL behavior

---

### 4. RL Model Extension

We modify the reward function:

From:
λ_inv * |inventory|

To:
λ_inv * inventory²

Result:

- More extreme price movements  
- Reduced liquidity provision  
- Higher return kurtosis  

---

## 📂 Repository Structure
