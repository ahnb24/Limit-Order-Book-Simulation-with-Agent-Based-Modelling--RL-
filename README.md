# 📊 Limit Order Book Simulation with Agent-Based Modelling

## 📌 Overview

This project implements an agent-based simulation of a financial market using a continuous double-auction Limit Order Book (LOB).

The goal is to study how interactions between Zero-Intelligence Traders (ZIT) and Reinforcement Learning (RL) traders affect market dynamics such as volatility, price formation, and profitability. The model is designed to reproduce key stylised facts observed in real financial markets, including fat-tailed returns and volatility clustering.

---

## ⚙️ Model Description

### 🏦 Limit Order Book (LOB)

The market operates as a continuous double auction:

$$
P_{\text{buy}} \geq \text{best ask}
$$

$$
P_{\text{sell}} \leq \text{best bid}
$$

Mid-price is defined as:

$$
M_t = \frac{\text{best bid}_t + \text{best ask}_t}{2}
$$

Orders are matched using price-time priority.

---

## 🤖 Agent Types

### 🔹 Zero-Intelligence Traders (ZIT)

- Submit random limit orders  
- Prices are drawn around a fundamental value  
- Provide liquidity to the market  

### 🔹 Reinforcement Learning Traders (RL)

- Use tabular Q-learning  
- Observe best bid and best ask  
- Choose actions: market buy, market sell, limit buy, limit sell, hold  

Reward function:

$$
R_t = \Delta PnL_t - \lambda_{\text{inv}} \cdot |inventory_t| - \lambda_{\text{hold}} \cdot \mathbf{1}_{\{hold\}}
$$

---

## 🔁 Simulation Process

At each time step:

1. Select a trader  
2. Observe the market state  
3. Take action (ZIT random / RL policy)  
4. Execute order  
5. Update inventory and cash  
6. Update Q-values (RL)  
7. Remove expired orders  

---

## 📊 Experiments & Results

### 1️⃣ Market Composition ($\omega$)

| $\omega$ | Excess Kurtosis | ACF Lag 1 | ACF Lag 2 | ACF Lag 3 | ACF Lag 4 |
|----------|----------------|----------|----------|----------|----------|
| 0.1 | 22.90 | 0.1021 | 0.0860 | 0.0028 | -0.0006 |
| 0.5 | 12.56 | 0.2292 | 0.1180 | 0.1168 | 0.1442 |
| 0.9 | 1.60  | 0.1760 | 0.0646 | 0.0322 | -0.0225 |

**Insights:**
- Low $\omega$ → extreme volatility and fat tails  
- Medium $\omega$ → strongest volatility clustering  
- High $\omega$ → stable, near-normal behavior  

---

### 📈 Mid Price Dynamics

![Mid Price ω=0.1](images/baseline_simulations/w01/actions.png)
![Mid Price ω=0.5](images/midprice_w05.png)
![Mid Price ω=0.9](images/midprice_w09.png)

- $\omega = 0.1$ → large jumps and inactivity  
- $\omega = 0.5$ → persistent deviations from fundamental  
- $\omega = 0.9$ → noisy but mean-reverting  

---

### ⚡ RL Action Frequencies

![Actions ω=0.1](images/actions_w01.png)
![Actions ω=0.5](images/actions_w05.png)
![Actions ω=0.9](images/actions_w09.png)

- Limit orders dominate at low $\omega$  
- Market orders increase with higher $\omega$  

---

### 📦 Inventory Dynamics

![Inventory ω=0.1](images/inventory_w01.png)
![Inventory ω=0.5](images/inventory_w05.png)
![Inventory ω=0.9](images/inventory_w09.png)

- RL inventories stay near zero when dominating  
- Mixed market → directional strategies emerge  
- High $\omega$ → symmetric random positions  

---

### 💰 PnL Dynamics

![PnL ω=0.1](images/pnl_w01.png)
![PnL ω=0.5](images/pnl_w05.png)
![PnL ω=0.9](images/pnl_w09.png)

- $\omega = 0.5$ → RL agents achieve profits  
- $\omega = 0.1$ → near zero-sum RL competition  
- $\omega = 0.9$ → noisy outcomes  

---

## 2️⃣ Parameter Sensitivity

### Inventory Penalty ($\lambda_{inv}$)

![Inventory Penalty](images/lambda_inv.png)

- $\lambda_{inv} = 0$ → speculative behavior  
- $\lambda_{inv} = 5$ → market-making behavior  

---

### Holding Penalty ($\lambda_{hold}$)

![Holding Penalty](images/lambda_hold.png)

- Higher $\lambda_{hold}$ → more trading  
- Increases volatility  

---

### Exploration Rate ($\epsilon_0$)

![Exploration Rate](images/epsilon.png)

- $\epsilon_0 = 0.1$ → stable learning  
- $\epsilon_0 = 0.9$ → noisy and unstable PnL  

---

## 3️⃣ Stylised Facts Calibration

| Metric | Baseline | Calibrated |
|--------|---------|-----------|
| Kurtosis | 12.56 | 3.20 |
| ACF Lag 1 | 0.2292 | 0.2972 |
| ACF Lag 2 | 0.1180 | 0.1614 |
| ACF Lag 3 | 0.1168 | 0.1710 |
| ACF Lag 4 | 0.1442 | 0.0835 |

The calibrated model successfully reproduces fat tails and volatility clustering observed in real markets.

---

## 4️⃣ RL Model Extension

Modified reward function:

$$
R_t = \Delta PnL_t - \lambda_{\text{inv}} \cdot (inventory_t)^2 - \lambda_{\text{hold}} \cdot \mathbf{1}_{\{hold\}}
$$

| Metric | Baseline | Modified |
|--------|---------|---------|
| Kurtosis | 12.56 | 16.65 |
| ACF Lag 1 | 0.2292 | 0.0734 |
| ACF Lag 2 | 0.1180 | 0.0720 |
| ACF Lag 3 | 0.1168 | 0.0306 |
| ACF Lag 4 | 0.1442 | 0.1023 |

**Insight:**
- Higher kurtosis → more extreme price movements  
- Lower ACF → weaker volatility persistence  

---

## 📂 Repository Structure
