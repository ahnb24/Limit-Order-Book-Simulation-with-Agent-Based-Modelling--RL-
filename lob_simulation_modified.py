import numpy as np
import random
from collections import deque, defaultdict
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import os



N = None                 # total agents
w = None               # fraction of ZIT agents; 1-w are RL agents
Pf = None              # fundamental price
S = None                # private value span (used by ZIT)
deltaP = None               # tick size
tc = None                # order expiry
TE = None            # total horizon
Pmin = None
Pmax = None
L = None                  # warm-start levels per side
D = None                   # depth per level
BURN = None
R = None                # runs

# RL hyperparameters
ALPHA = None
GAMMA = None
EPS0 = None
EPS_DECAY = None
EPS_MIN = None

# RL actions
# 0: BUY MO, 1: SELL MO, 2: PLACE LIMIT at best_bid + delta_l,
# 3: PLACE LIMIT at best_ask - delta_l, 4: HOLD
delta_l = None

# Reward penalties
LAMBDA_INV =  None
LAMBDA_HOLD = None


# -------------------------------
# DATA STRUCTURES
# -------------------------------
class Order:
    def __init__(self, trader_id, price, side, expiry, time):
        self.trader_id = trader_id
        self.price = price
        self.side = side
        self.expiry = expiry
        self.time = time

class LimitOrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}

    def add_order(self, order):
        book = self.bids if order.side == "buy" else self.asks
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)

    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    def remove_expired(self, t):
        for book in [self.bids, self.asks]:
            for price in list(book.keys()):
                book[price] = deque([o for o in book[price] if o.expiry > t])
                if not book[price]:
                    del book[price]

    def match_order(self, order):
        trades = []
        if order.side == "buy":
            best_ask = self.best_ask()
            if best_ask is not None and order.price >= best_ask:
                resting = self.asks[best_ask].popleft()
                trades.append((best_ask, resting))
                if not self.asks.get(best_ask):
                    self.asks.pop(best_ask, None)
            else:
                self.add_order(order)
        else:  # sell
            best_bid = self.best_bid()
            if best_bid is not None and order.price <= best_bid:
                resting = self.bids[best_bid].popleft()
                trades.append((best_bid, resting))
                if not self.bids.get(best_bid):
                    self.bids.pop(best_bid, None)
            else:
                self.add_order(order)
        return trades

# -------------------------------
# WARM-START BOOK
# -------------------------------
def warm_start(book, t0=0):
    for i in range(1, L + 1):
        for _ in range(D):
            book.add_order(Order(-1, Pf - i * deltaP, "buy",  t0 + tc, t0))
        for _ in range(D):
            book.add_order(Order(-1, Pf + i * deltaP, "sell", t0 + tc, t0))

# -------------------------------
# PORTFOLIOS
# -------------------------------
class Portfolio:
    def __init__(self):
        self.cash = 0.0
        self.inv  = 0
    def pnl(self, mid):
        return self.cash + self.inv * (mid if mid is not None else 0.0)

def init_portfolios(agent_ids):
    return {i: Portfolio() for i in agent_ids}

# -------------------------------
# AGENTS
# -------------------------------
def build_agent_partitions():
    zit_ids = {i for i in range(N) if np.random.rand() < w}
    rl_ids  = set(range(N)) - zit_ids

    zit_side_by_agent = [None] * N
    for i in zit_ids:
        zit_side_by_agent[i] = "buy" if np.random.rand() < 0.5 else "sell"

    return zit_ids, rl_ids, zit_side_by_agent


# -------------------------------
# EXECUTION ACCOUNTING
# -------------------------------
def apply_execution_for_incoming(portfolios, trader_id, side, trade_price):
    pf = portfolios.get(trader_id)
    if pf is None: return
    if side == "buy":
        pf.inv  += 1
        pf.cash -= trade_price
    else:
        pf.inv  -= 1
        pf.cash += trade_price

def apply_execution_for_resting(portfolios, resting_order, trade_price):
    pf = portfolios.get(resting_order.trader_id)
    if pf is None: return
    if resting_order.side == "sell":
        pf.inv  -= 1
        pf.cash += trade_price
    else:
        pf.inv  += 1
        pf.cash -= trade_price

# -------------------------------
# RL POLICY
# -------------------------------
Q = defaultdict(float)

def epsilon_greedy(state, eps):
    if random.random() < eps:
        return random.randrange(5)
    vals = [Q[(state, a)] for a in range(5)]
    m = max(vals)
    best = [i for i,v in enumerate(vals) if v == m]
    return random.choice(best)

def q_update(s, a, r, s_next, alpha=ALPHA, gamma=GAMMA):
    max_next = max(Q[(s_next, ap)] for ap in range(5))
    Q[(s,a)] += alpha * (r + gamma * max_next - Q[(s,a)])

# -------------------------------
# SIMULATION RUN
# -------------------------------
def run_sim(seed):
    np.random.seed(seed)
    random.seed(seed)
    Q.clear()

    book = LimitOrderBook()
    warm_start(book, t0=0)
    last_mid = Pf
    
    zit_ids, rl_ids, zit_side_by_agent = build_agent_partitions()
    zit_portfolios = init_portfolios(zit_ids)
    rl_portfolios  = init_portfolios(rl_ids)

    eps = EPS0

    mids, inv_rl, inv_zit, pnl_rl, pnl_zit = [], [], [], [], []
    action_counts = np.zeros(5, dtype=float)

    for t in range(1, TE+1):
        trader_id = t % N

        # housekeeping
        book.remove_expired(t)
        s = (book.best_bid(), book.best_ask())

        best_bid, best_ask = book.best_bid(), book.best_ask()
        if best_bid is not None and best_ask is not None:
            last_mid = (best_bid + best_ask)/2

        is_rl = (trader_id in rl_ids)
        if is_rl:
            pnl_before = rl_portfolios[trader_id].pnl(last_mid)

        # ----- ZIT trader -----
        if trader_id in zit_ids:
            side_i = zit_side_by_agent[trader_id] 
            price  = zit_limit_price(side_i)
            order  = Order(trader_id, price, side_i, t + tc, t)
            trades = book.match_order(order)
            for trade_price, resting in trades:
                # Incoming ZIT
                apply_execution_for_incoming(zit_portfolios, trader_id, order.side, trade_price)
                # Resting counterparty (could be ZIT or RL) — update the correct group once
                if resting.trader_id in zit_portfolios:
                    apply_execution_for_resting(zit_portfolios, resting, trade_price)
                elif resting.trader_id in rl_portfolios:
                    apply_execution_for_resting(rl_portfolios, resting, trade_price)

        # ----- RL trader -----
        else:
            a = epsilon_greedy(s, eps)
            action_counts[a] += 1

            if a == 0:
                price, side_ = Pmax, "buy"       # Buy MO
            elif a == 1:
                price, side_ = Pmin, "sell"      # Sell MO
            elif a == 2:
                bb = book.best_bid()
                price, side_ = (bb if bb is not None else Pf) + delta_l, "buy"
            elif a == 3:
                ba = book.best_ask()
                price, side_ = (ba if ba is not None else Pf) - delta_l, "sell"
            else:
                price, side_ = None, None        # Hold

            if a < 4:
                px = int(price // deltaP) * deltaP if side_=="buy" else int(np.ceil(price/deltaP))*deltaP
                order = Order(trader_id, px, side_, t+tc, t)
                trades = book.match_order(order)
                for trade_price, resting in trades:
                    # Incoming RL
                    apply_execution_for_incoming(rl_portfolios, trader_id, order.side, trade_price)
                    # Resting counterparty — update the correct group once
                    if resting.trader_id in rl_portfolios:
                        apply_execution_for_resting(rl_portfolios, resting, trade_price)
                    elif resting.trader_id in zit_portfolios:
                        apply_execution_for_resting(zit_portfolios, resting, trade_price)

        # update mid
        best_bid, best_ask = book.best_bid(), book.best_ask()
        if best_bid is not None and best_ask is not None:
            last_mid = (best_bid + best_ask)/2
        mids.append(last_mid)

        # RL learning update
        if is_rl:
            pnl_after = rl_portfolios[trader_id].pnl(last_mid)
            r = (pnl_after - pnl_before) - LAMBDA_INV * (rl_portfolios[trader_id].inv**2)
            if a == 4:
                r -= LAMBDA_HOLD
            s_next = (book.best_bid(), book.best_ask())
            q_update(s, a, r, s_next, alpha=ALPHA, gamma=GAMMA)

        # epsilon decay
        if t % 1000 == 0:
            eps = max(EPS_MIN, eps * EPS_DECAY)

        # group aggregates (real portfolios)
        inv_rl.append(np.mean([p.inv for p in rl_portfolios.values()]) if rl_portfolios else 0.0)
        pnl_rl.append(np.mean([p.pnl(last_mid) for p in rl_portfolios.values()]) if rl_portfolios else 0.0)
        inv_zit.append(np.mean([p.inv for p in zit_portfolios.values()]) if zit_portfolios else 0.0)
        pnl_zit.append(np.mean([p.pnl(last_mid) for p in zit_portfolios.values()]) if zit_portfolios else 0.0)

    # burn-in
    mids = mids[BURN:]
    inv_rl, inv_zit = inv_rl[BURN:], inv_zit[BURN:]
    pnl_rl, pnl_zit = pnl_rl[BURN:], pnl_zit[BURN:]

    # stylised facts
    returns = [np.log(mids[k] / mids[k-100]) for k in range(100, len(mids), 100)]
    returns = np.array(returns)
    excess_kurt = kurtosis(returns, fisher=True) if len(returns) > 0 else np.nan
    acfs = acf(returns**2, nlags=4, fft=True)[1:] if len(returns) > 10 else np.array([np.nan]*4)

    return {
        "excess_kurt": excess_kurt,
        "acfs": acfs,
        "mids": mids,
        "inv_rl": inv_rl,
        "inv_zit": inv_zit,
        "pnl_rl": pnl_rl,
        "pnl_zit": pnl_zit,
        "action_counts": action_counts,
    }

# -------------------------------
# ZIT BEHAVIOR
# -------------------------------
def zit_limit_price(side):
    """
    Returns a random limit price for ZIT traders based on their private value.
    Buyers sample private values above Pf and post below that; 
    sellers sample private costs below Pf and post above that.
    """
    if side == "buy":
        v = np.random.uniform(Pf, Pf + S)
        price = np.random.uniform(Pmin, v)
        price = int(price // deltaP) * deltaP
    else:  # sell
        c = np.random.uniform(Pf - S, Pf)
        price = np.random.uniform(c, Pmax)
        price = int(np.ceil(price / deltaP)) * deltaP
    return price


# -------------------------------
# MULTIPLE RUNS
# -------------------------------
def multiple_runs(parameters):
    global N, w, Pf, S, deltaP, tc, TE, Pmin, Pmax, L, D, BURN, R
    global ALPHA, GAMMA, EPS0, EPS_DECAY, EPS_MIN
    global delta_l, LAMBDA_INV, LAMBDA_HOLD

    save_path = parameters['save_path']
    
    N = parameters['N']                 # total agents
    w = parameters['w']               # fraction of ZIT agents; 1-w are RL agents
    Pf = parameters['Pf']              # fundamental price
    S = parameters['S']                # private value span (used by ZIT)
    deltaP = parameters['deltaP']               # tick size
    tc = parameters['tc']                # order expiry
    TE = parameters['TE']            # total horizon
    Pmin = deltaP
    Pmax = 2 * (Pf + S)
    L = parameters['L']                  # warm-start levels per side
    D = parameters['D']                   # depth per level
    BURN = int(round(0.1 * TE))
    R = parameters['R']                # runs

    # RL hyperparameters
    ALPHA = parameters['ALPHA']
    GAMMA = parameters['GAMMA']
    EPS0 = parameters['EPS0']
    EPS_DECAY = parameters['EPS_DECAY']
    EPS_MIN = parameters['EPS_MIN']

    # RL actions
    # 0: BUY MO, 1: SELL MO, 2: PLACE LIMIT at best_bid + delta_l,
    # 3: PLACE LIMIT at best_ask - delta_l, 4: HOLD
    delta_l = 1 * deltaP

    # Reward penalties
    LAMBDA_INV =  parameters['LAMBDA_INV']
    LAMBDA_HOLD = parameters['LAMBDA_HOLD']


    results = []
    for r in range(R):
        print(f"Starting run {r+1} of {R}...")
        results.append(run_sim(seed=r))

    def pad_to_max(arrs):
        maxlen = max(len(a) for a in arrs)
        return np.array([np.pad(a, (0, maxlen-len(a)), constant_values=np.nan) for a in arrs])

    avg_mid    = np.nanmean(pad_to_max([res["mids"]     for res in results]), axis=0)
    avg_inv_rl = np.nanmean(pad_to_max([res["inv_rl"]   for res in results]), axis=0)
    avg_inv_zit= np.nanmean(pad_to_max([res["inv_zit"]  for res in results]), axis=0)
    avg_pnl_rl = np.nanmean(pad_to_max([res["pnl_rl"]   for res in results]), axis=0)
    avg_pnl_zit= np.nanmean(pad_to_max([res["pnl_zit"]  for res in results]), axis=0)

    total_actions = np.sum([res["action_counts"] for res in results], axis=0)
    total_actions = total_actions / total_actions.sum() if total_actions.sum() > 0 else total_actions

    avg_kurt = np.nanmean([res["excess_kurt"] for res in results])
    avg_acf  = np.nanmean(np.stack([res["acfs"] for res in results]), axis=0)

    print("\nExcess kurtosis:", avg_kurt)
    for lag, val in enumerate(avg_acf[:4], start=1):
        print(f"ACF squared returns lag {lag}: {val:.4f}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/stats.txt", "w") as f:
            f.write(f"Excess kurtosis: {avg_kurt}\n")
            
            for lag, val in enumerate(avg_acf[:4], start=1):
                f.write(f"ACF squared returns lag {lag}: {val:.4f}\n")
    
    # -------------------------------
    # PLOTS
    # -------------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        x = np.arange(BURN, BURN + len(avg_mid))

        # --- Mid Price ---
        fig, ax = plt.subplots()
        ax.plot(x, avg_mid)
        ax.axhline(Pf, color="red", linestyle="--", label="Fundamental Price")
        ax.set_title("Average Mid Price (post burn-in)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mid Price")
        ax.set_xlim(x[0], x[-1])
        ax.margins(x=0, y=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{save_path}/mid_price.png", dpi=300)
        plt.close(fig)

        # --- Inventory ---
        x = np.arange(BURN, BURN + len(avg_inv_rl))
        fig, ax = plt.subplots()
        ax.plot(x, avg_inv_rl, label="RL Inventory")
        ax.plot(x, avg_inv_zit, label="ZIT Inventory")
        ax.set_title("Average Inventory per Tick (post burn-in)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Inventory")
        ax.set_xlim(x[0], x[-1])
        ax.margins(x=0, y=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{save_path}/inventory.png", dpi=300)
        plt.close(fig)

        # --- PnL ---
        x = np.arange(BURN, BURN + len(avg_pnl_rl))
        fig, ax = plt.subplots()
        ax.plot(x, avg_pnl_rl, label="RL PnL")
        ax.plot(x, avg_pnl_zit, label="ZIT PnL")
        ax.set_title("Average PnL per Tick (post burn-in)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("PnL")
        ax.set_xlim(x[0], x[-1])
        ax.margins(x=0, y=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{save_path}/pnl.png", dpi=300)
        plt.close(fig)

        # --- RL Action Frequencies ---
        fig, ax = plt.subplots()
        ax.bar(range(5), total_actions)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["BuyMO", "SellMO", "Bid+δ", "Ask−δ", "Hold"])
        ax.set_title("RL Action Frequencies (post burn-in)")
        fig.tight_layout()
        fig.savefig(f"{save_path}/actions.png", dpi=300)
        plt.close(fig)


    metrics = {
        "avg_kurt": avg_kurt,
        "acf_lag1": avg_acf[0],
        "acf_lag2": avg_acf[1],
        "acf_lag3": avg_acf[2],
        "acf_lag4": avg_acf[3],
    }

    return metrics


