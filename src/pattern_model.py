import os
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -------------------------------
# SEED CONTROL (NEW)
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = int(os.environ.get("SEED", 3))

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "../data"
MODEL_DIR = "../models"

H = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Utils
# -------------------------------
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    train = np.load(f"{DATA_DIR}/train_windows.npy")
    train_price = np.load(f"{DATA_DIR}/train_price_windows.npy")

    test = np.load(f"{DATA_DIR}/test_windows.npy")
    test_price = np.load(f"{DATA_DIR}/test_price_windows.npy")

    return train, train_price, test, test_price


# -------------------------------
# Feature Engineering
# -------------------------------
def engineer_features(windows):
    close = windows[:, :, 3]
    volume = windows[:, :, 4]
    high = windows[:, :, 1]
    low = windows[:, :, 2]

    returns = close[:, 1:] - close[:, :-1]
    vol = np.std(returns, axis=1, keepdims=True)

    momentum = (close[:, -1] - close[:, 0]).reshape(-1, 1)
    vol_norm = (volume[:, -1] / (np.mean(volume, axis=1) + 1e-8)).reshape(-1, 1)

    tr1 = high[:, 1:] - low[:, 1:]
    tr2 = np.abs(high[:, 1:] - close[:, :-1])
    tr3 = np.abs(low[:, 1:] - close[:, :-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(tr, axis=1, keepdims=True)

    delta = close[:, 1:] - close[:, :-1]
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)

    rs = np.mean(gain, axis=1, keepdims=True) / (np.mean(loss, axis=1, keepdims=True) + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    ma = np.mean(close, axis=1, keepdims=True)
    std = np.std(close, axis=1, keepdims=True)
    bb_pos = (close[:, -1].reshape(-1, 1) - ma) / (std + 1e-8)

    X = returns.reshape(returns.shape[0], -1)

    return np.hstack([X, vol, momentum, vol_norm, atr, rsi, bb_pos])


# -------------------------------
# Forward Returns
# -------------------------------
def compute_forward_returns(windows):
    close = windows[:, :, 3]
    n = len(close)

    fr = []

    for i in range(n):
        if i + H >= n:
            break

        curr_price = close[i, -1]
        future_price = close[i + H, -1]

        r = (future_price / (curr_price + 1e-8)) - 1.0
        r = np.clip(r, -0.05, 0.05)

        fr.append(r)

    return np.array(fr)

# ===============================
# ONLY CHANGED PART: TradingEnv
# ===============================

# (ONLY showing modified part clearly — rest remains same)

class TradingEnv:
    def __init__(self, X, returns, cost=0.0001):
        self.X = X
        self.returns = returns
        self.cost = cost
        self.n = len(X)

    def reset(self):
        self.t = 0
        self.position = 0
        self.equity = 1.0
        self.peak = 1.0
        self.bars_in_trade = 0
        return self._get_state()

    def _get_state(self):
        start_10 = max(0, self.t - 10)
        start_50 = max(0, self.t - 50)

        recent = self.X[start_10:self.t+1]
        long = self.X[start_50:self.t+1]

        vol_short = np.std(recent) if len(recent) > 1 else 0.0
        vol_long = np.std(long) if len(long) > 1 else 1.0
        vol_ratio = vol_short / (vol_long + 1e-8)

        close_series = self.X[start_50:self.t+1, 0]
        mean = np.mean(close_series) if len(close_series) > 1 else 0.0
        std = np.std(close_series) if len(close_series) > 1 else 1.0
        close_z = (self.X[self.t][0] - mean) / (std + 1e-8)

        if self.t > 0:
            unrealised_pnl = self.position * self.returns[self.t - 1]
        else:
            unrealised_pnl = 0.0

        # 🔥 STABILITY FIX
        unrealised_pnl = np.clip(unrealised_pnl, -0.02, 0.02)

        drawdown = (self.peak - self.equity) / self.peak

        return np.concatenate([
            self.X[self.t],
            [
                self.position,
                unrealised_pnl,
                self.bars_in_trade / 20,
                drawdown,
                vol_ratio,
                close_z
            ]
        ])
    def step(self, action):
        prev_pos = self.position
        self.position = int(action)

        cost = self.cost if prev_pos != self.position else 0.0

        # 🔥 flip penalty (NEW)
        flip_penalty = 0.0005 if prev_pos != self.position else 0.0

        r = self.returns[self.t]

        # -------------------------------
        # REAL TRADING PNL (for equity)
        # -------------------------------
        pnl = self.position * r - cost - flip_penalty

        # update equity ONLY with real pnl
        self.equity *= (1 + pnl)
        self.equity = max(self.equity, 1e-6)

        # update peak
        self.peak = max(self.peak, self.equity)

        # compute drawdown
        drawdown = (self.peak - self.equity) / self.peak

        # -------------------------------
        # LEARNING REWARD (separate)
        # -------------------------------
        effective_pnl = pnl

        # inactivity bonus
        reward = effective_pnl
        if self.position == 0:
            reward += 0.0002

        # normalize + stabilize
        reward = reward / (0.05 + abs(effective_pnl))
        reward = np.tanh(reward)

        # risk control — convex penalty for deep drawdowns
        reward -= 0.08 * (drawdown ** 1.2)

        # behavior penalties
        reward -= 0.0035 * abs(self.position)
        reward -= 0.0015 * (self.bars_in_trade ** 1.2)

        # clip
        reward = np.clip(reward, -0.08, 0.08)

        # -------------------------------
        # bookkeeping
        # -------------------------------
        if self.position == 1:
            self.bars_in_trade += 1
        else:
            self.bars_in_trade = 0

        self.t += 1
        done = self.t >= self.n - 1

        return self._get_state(), reward, done

# -------------------------------
# Q Network
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)
# ===============================
# ONLY CHANGE IN AGENT
# ===============================

class DQNAgent:
    def __init__(self, state_dim):
        self.model = QNetwork(state_dim).to(DEVICE)
        self.target = QNetwork(state_dim).to(DEVICE)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.memory = deque(maxlen=50000)
        self.batch_size = 128  # updated

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.992
        self.epsilon_min = 0.05

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, 1)

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            return torch.argmax(self.model(state), dim=1).item()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        s, a, r, ns, d = zip(*batch)

        s = torch.from_numpy(np.array(s)).float().to(DEVICE)
        ns = torch.from_numpy(np.array(ns)).float().to(DEVICE)
        r  = torch.from_numpy(np.array(r)).float().to(DEVICE)
        a  = torch.from_numpy(np.array(a)).long().to(DEVICE)
        d  = torch.from_numpy(np.array(d)).float().to(DEVICE)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target(ns).max(1)[0]

        target = r + self.gamma * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# -------------------------------
# Metrics
# -------------------------------
def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(8760)


def compute_mdd(equity_curve):
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return np.max(drawdown)


# -------------------------------
# NEW BASELINES
# -------------------------------
def evaluate_always_long(returns):
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    return np.array(equity)


def evaluate_always_flat(returns):
    return np.ones(len(returns) + 1)


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_agent(agent, X, returns, label="Agent"):
    env = TradingEnv(X, returns)
    state = env.reset()

    equity_curve = [1.0]
    actions = []

    while True:
        action = agent.act(state, epsilon=0.0)
        state, reward, done = env.step(action)

        equity_curve.append(env.equity)
        actions.append(action)

        if done:
            break

    equity_curve = np.array(equity_curve)
    step_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)

    sharpe = compute_sharpe(step_returns)
    mdd = compute_mdd(equity_curve)

    print(f"\n===== {label} =====")
    print(f"Final Equity: {equity_curve[-1]:.4f}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Long %: {np.mean(actions):.2%}")  # 🔥 NEW

    return equity_curve


# -------------------------------
# MAIN
# -------------------------------
def main():
    set_seed(SEED)
    print("Using device:", DEVICE)

    ensure_dirs()

    train_windows, train_price, test_windows, test_price = load_data()

    print(f"[data] train_windows: {train_windows.shape}  train_price: {train_price.shape}")
    print(f"[data] test_windows:  {test_windows.shape}   test_price:  {test_price.shape}")

    X_train = engineer_features(train_windows)
    X_test = engineer_features(test_windows)

    train_returns = compute_forward_returns(train_price)
    test_returns = compute_forward_returns(test_price)

    X_train = X_train[:len(train_returns)]
    X_test = X_test[:len(test_returns)]

    scaler = StandardScaler()
    scaler.fit(X_train[:int(0.7 * len(X_train))])

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    env = TradingEnv(X_train, train_returns)
    agent = DQNAgent(len(env.reset()))

    print("\n🚀 Training...\n")

    EPISODES = 200
    for ep in tqdm(range(EPISODES)):
        state = env.reset()

        while True:
            action = agent.act(state)
            ns, r, d = env.step(action)
            agent.remember(state, action, r, ns, d)

            # ✅ Balanced replay (fixes slowdown + keeps learning strong)
            if len(agent.memory) > 5000 and random.random() < 0.2:
                agent.replay()

            state = ns
            if d:
                break

        # episode-level updates
        agent.decay_epsilon()
        agent.soft_update()

    print("\n===== EVALUATION =====")

    agent_curve = evaluate_agent(agent, X_test, test_returns, "DQN")

    random_curve = evaluate_always_long(np.random.permutation(test_returns))
    print(f"Shuffle Test Equity: {random_curve[-1]:.4f}")

    long_curve = evaluate_always_long(test_returns)
    print(f"Always Long Equity: {long_curve[-1]:.4f}")

    flat_curve = evaluate_always_flat(test_returns)
    print(f"Always Flat Equity: {flat_curve[-1]:.4f}")


if __name__ == "__main__":
    main()