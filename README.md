# 📊 Limit Order Book Simulation with Agent-Based Modelling (RL)

## 📌 Overview

In this project the aim is to study the interaction between Zero-Intelligence Traders (ZIT) and Reinforcement Learning (RL) traders within an agent-based artificial financial market.The trading environment is a continuous double-auction limit order book (LOB). Using the lob_simulation.ipynb notebook, the task is to analyse how the presence and behaviour of different agent types influence market outcomes and to compare the results with stylised facts observed in real financial data.The components and structure of the artificial financial market model are described below.
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
<p align="center">
  <img src="baseline_simulations/w01/mid_price.png" width="500"><br>
  <em>ω = 0.1 → large jumps and inactivity</em>
</p>
<p align="center">
  <img src="baseline_simulations/w05/mid_price.png" width="500"><br>
  <em>ω = 0.5 → persistent deviations from fundamental</em>
</p>
<p align="center">
  <img src="baseline_simulations/w09/mid_price.png" width="500"><br>
  <em>ω = 0.9 → noisy but mean-reverting </em>
</p>
 
---

### ⚡ RL Action Frequencies

<p align="center">
  <img src="baseline_simulations/w01/actions.png" width="500"><br>
  <em>ω = 0.1 </em>
</p>
<p align="center">
  <img src="baseline_simulations/w05/actions.png" width="500"><br>
  <em>ω = 0.5 </em>
</p>
<p align="center">
  <img src="baseline_simulations/w09/actions.png" width="500"><br>
  <em>ω = 0.9 </em>
</p>

- Limit orders dominate at low $\omega$  
- Market orders increase with higher $\omega$  

---

### 📦 Inventory Dynamics

<p align="center">
  <img src="baseline_simulations/w01/inventory.png" width="500"><br>
  <em>ω = 0.1 </em>
</p>
<p align="center">
  <img src="baseline_simulations/w05/inventory.png" width="500"><br>
  <em>ω = 0.5 </em>
</p>
<p align="center">
  <img src="baseline_simulations/w09/inventory.png" width="500"><br>
  <em>ω = 0.9 </em>
</p>

- RL inventories stay near zero when dominating  
- Mixed market → directional strategies emerge  
- High $\omega$ → symmetric random positions  

---

### 💰 PnL Dynamics

<p align="center">
  <img src="baseline_simulations/w01/pnl.png" width="500"><br>
  <em>ω = 0.1 → near zero-sum RL competition</em>
</p>
<p align="center">
  <img src="baseline_simulations/w05/pnl.png" width="500"><br>
  <em>ω = 0.5 → RL agents achieve profits</em>
</p>
<p align="center">
  <img src="baseline_simulations/w09/pnl.png" width="500"><br>
  <em>ω = 0.9 → noisy outcomes</em>
</p>


---

## 2️⃣ Parameter Sensitivity

### Inventory Penalty ($\lambda_{inv}$)

<p align="center">
  <img src="parameters_sensetivity/lambda_inv0/inventory.png" width="500"><br>
  <em>λ<sub>inv</sub> = 0 → speculative behavior</em>
</p>
<p align="center">
  <img src="parameters_sensetivity/lambda_inv05/inventory.png" width="500"><br>
  <em>λ<sub>inv</sub> = 5 → market-making behavior</em>
</p>

---

### Holding Penalty ($\lambda_{hold}$)

<p align="center">
  <img src="parameters_sensetivity/lambda_hold0/actions.png" width="500"><br>
  <em>λ<sub>inv</sub> = 0 </em>
</p>
<p align="center">
  <img src="parameters_sensetivity/lambda_hold05/actions.png" width="500"><br>
  <em>λ<sub>inv</sub> = 5 </em>
</p>

- Higher $\lambda_{hold}$ → more trading  
- Increases volatility  

---

### Exploration Rate ($\epsilon_0$)

<p align="center">
  <img src="parameters_sensetivity/eps0-01/pnl.png" width="500"><br>
  <em>λ<sub>inv</sub> = 0 → stable learning </em>
</p>
<p align="center">
  <img src="parameters_sensetivity/eps0-09/pnl.png" width="500"><br>
  <em>λ<sub>inv</sub> = 0.9 → noisy and unstable PnL</em>
</p>


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

