


```md
# DQN Trading System

## Overview
A Deep Q-Network (DQN) based trading agent that learns profitable strategies from historical OHLCV data using reinforcement learning. The system incorporates risk-aware reward shaping to balance profit generation with drawdown control and trading discipline.

---

## Objective
- Learn trading behavior without explicit rules
- Maximize returns while controlling:
  - Drawdown
  - Overtrading
  - Instability

---

## System Architecture

Raw OHLCV Data  
→ Windowing (20 steps)  
→ Feature Engineering (returns, ATR, RSI, volatility, etc.)  
→ StandardScaler (train-only fit)  
→ Trading Environment  
→ DQN Agent (Q-Network + Target Network)  
→ Experience Replay  
→ Evaluation (Baselines)

---

## Feature Engineering

Key features derived from OHLCV windows:
- Returns
- Volatility
- Momentum
- Volume Normalization
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Bollinger Band Position

---

## Trading Environment

### State
- Market features (engineered signals)
- Agent context:
  - Position
  - Unrealized PnL
  - Bars in trade
  - Drawdown
  - Volatility ratio

### Actions
- `0` → Flat
- `1` → Long

---

## Reward Function

Reward is carefully shaped to balance profit and risk:

- Base: trading PnL
- Inactivity bonus (encourage staying flat)
- Normalization: `tanh`
- Convex drawdown penalty
- Holding penalty
- Duration penalty
- Final clipping for stability

---

## Model Architecture



Input → Linear(256) → ReLU
→ Linear(128) → ReLU
→ Linear(2)   → Q-values



- Target network (soft updates)
- Experience replay buffer
- Huber loss
- Gradient clipping

---

## Training Setup

- Episodes: ~200
- Replay: probabilistic (~20%)
- Discount factor: 0.99
- Epsilon-greedy exploration
- Forward returns: H = 5

---

## Evaluation Metrics

- Final Equity
- Sharpe Ratio
- Max Drawdown
- Long %

### Baselines
- Always Long
- Always Flat
- Shuffle Test (leakage check)

---

## Results

| Metric | DQN |
|-------|-----|
| Final Equity | ~2.4 – 3.4x |
| Sharpe | ~1.7 – 2.3 |
| Max Drawdown | ~70 – 80% |
| Long % | ~50 – 60% |

---

## Key Insights

### What Worked
- Learned consistent trading signal
- Strong performance vs random baseline
- Stable across multiple seeds
- No data leakage observed

### Limitations
- High drawdown (~80%)
- No position sizing
- No stop-loss / take-profit
- Discrete action space
- No temporal memory

---

## Engineering Challenges

- Reward tuning instability
- Replay vs speed tradeoff
- Target network oscillation
- Drawdown control limitations

---

## Conclusion

This project demonstrates that reinforcement learning can extract meaningful trading signals from raw market data. However, structural limitations of DQN (discrete actions, no risk control mechanisms) prevent production-level performance.

---

## Next Steps

- PPO (Proximal Policy Optimization)
- Continuous action space (position sizing)
- LSTM / Transformer for temporal modeling
- Risk-based objectives (Sharpe / Calmar)

---

## Run the Project

```bash
python src/pattern_model.py
````

---

## Final Note

This is a **research prototype**, not a production trading system.

---

````

---

# 🧠 Why this is better than raw doc

| Raw Doc | This README |
|--------|-----------|
| Too long | Scannable |
| Hard to read | Structured |
| Academic | Engineering-focused |
| Not GitHub friendly | Perfect for GitHub |

---

# 🔥 Final Verdict

👉 **YES — this is the correct Option 1 implementation**  
👉 Your repo now becomes:

```text
✔ Clean code
✔ Strong ML logic
✔ Clear explanation
✔ Interview-ready
✔ Portfolio-grade
````

---

