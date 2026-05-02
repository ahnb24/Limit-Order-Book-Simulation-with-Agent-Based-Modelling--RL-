# 📊 Limit Order Book Simulation with Agent-Based Modelling

## 📌 Overview

This project implements an **agent-based financial market simulation** using a continuous double-auction **Limit Order Book (LOB)**.

The goal is to study how interactions between:
- **Zero-Intelligence Traders (ZIT)**
- **Reinforcement Learning Traders (RL)**

affect market dynamics such as:
- Volatility
- Price formation
- Profitability

---

## ⚙️ Model Description

### 🏦 Limit Order Book (LOB)

The market follows a continuous double auction:

$$
P_{\text{buy}} \geq \text{best ask}
$$

$$
P_{\text{sell}} \leq \text{best bid}
$$

Mid-price:

$$
M_t = \frac{\text{best bid}_t + \text{best ask}_t}{2}
$$

Orders are matched using **price-time priority**.

---

## 🤖 Agent Types

### 🔹 Zero-Intelligence Traders (ZIT)

- Submit random limit orders  
- Prices drawn around fundamental value  
- Provide liquidity  

---

### 🔹 Reinforcement Learning Traders (RL)

- Use **Q-learning**
- Observe LOB state (best bid/ask)
- Choose actions:
  - Market Buy
  - Market Sell
  - Limit Buy
  - Limit Sell
  - Hold

Reward function:

$$
R_t = \Delta PnL_t - \lambda_{\text{inv}} \cdot |inventory_t| - \lambda_{\text{hold}} \cdot \mathbf{1}_{\{hold\}}
$$

---

## 🔁 Simulation Process

At each time step:

1. Select trader  
2. Observe state  
3. Take action (ZIT or RL)  
4. Execute order  
5. Update:
   - Inventory  
   - Cash  
   - Q-values (RL only)  
6. Remove expired orders  

---

## 📊 Experiments & Results

---

### 1️⃣ Market Composition (ω)

| ω | Excess Kurtosis | ACF Lag 1 | ACF Lag 2 | ACF Lag 3 | ACF Lag 4 |
|--|--|--|--|--|--|
| 0.1 | 22.90 | 0.1021 | 0.0860 | 0.0028 | -0.0006 |
| 0.5 | 12.56 | 0.2292 | 0.1180 | 0.1168 | 0.1442 |
| 0.9 | 1.60  | 0.1760 | 0.0646 | 0.0322 | -0.0225 |

**Key Insight:**
- Low ω → extreme volatility (fat tails)
- Medium ω → strongest volatility clustering
- High ω → stable, near-normal behavior :contentReference[oaicite:0]{index=0}  

---

### 📈 Mid Price Dynamics

```markdown
![Mid Price ω=0.1](images/midprice_w01.png)
![Mid Price ω=0.5](images/midprice_w05.png)
![Mid Price ω=0.9](images/midprice_w09.png)
