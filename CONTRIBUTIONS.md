# Contributions & Differences vs the FlappyQ paper

This project is a **beginner-first**, **tabular** Q-learning implementation of Flappy Bird.
It follows the *spirit* of the “Applying Q-Learning to Flappy Bird” paper (Ebeling‑Rump et al.)
and connects the *code decisions* to the convergence conditions discussed in
Regehr & Ayoub (2021), “An Elementary Proof that Q-learning Converges Almost Surely”.

---

## 1) What I have changed / added (vs the paper)

1. **Clean environment wrapper (MDP API)**  
   I use **PLE (PyGame Learning Environment)** `FlappyBird` and wrap it in `env_simple.py`
   to expose a tiny `reset()` / `step(action)` interface. This makes the “MDP” structure explicit:
   state → action → (reward, next state, terminal).

2. **State features kept simple and interpretable**  
   I represent each moment in the game using three quantities:
   - `x`: distance to the next pipe
   - `y`: vertical offset (`gap_top_y - bird_y`)
   - `v`: vertical velocity  
   Then I discretise them into bins so the state space is **finite** and the Q-table is inspectable.

3. **Reward shaping that matches the paper’s intent**  
   I use a dense survival reward:
   - **+15** per frame alive
   - **−1000** on crash  
   This strongly correlates with “survive longer ⇒ pass more pipes” and improves sample efficiency
   for a purely tabular method.

4. **Diagnostics (learning curves + state visitation heatmap)**  
   The paper focuses on the algorithm; I add plots that show:
   - how performance evolves (frames/pipes over episodes)
   - which discretised states are actually visited  
   This makes it much easier to debug and to understand why learning is noisy.

5. **Project structure for learning**  
   Files are split by concept:
   - environment (`env_simple.py`)
   - agent (`agent_simple.py`)
   - training script (`train.py`)
   - analysis (`plot.py`)
   - evaluation/playback (`watch.py`)  
   so a beginner can read one idea at a time.

---

## 2) What remains the same in spirit

- **Tabular Q-learning** with a Q-table over discretised states.
- **ε-greedy** exploration.
- The goal is still “learn a policy that survives longer and passes pipes”.

---

## 3) Mathematics & convergence: how the code lines up with the theory

### 3.1 The update rule (Watkins Q-learning)

In `agent_simple.py` I implement the standard update:


$$
Q \leftarrow Q + \alpha \,(\text{target} - Q)
\qquad\text{where}\qquad
\text{target} = r + \gamma\max_{a'}Q(s',a').
$$

### 3.2 Why I discretise (finite MDP)

Regehr & Ayoub’s proof (and classical proofs) assume a **finite** state and action space.
Flappy Bird is continuous, so I discretise $(x, y, v)$ into bins to create a finite MDP.
That’s why the Q-table is a fixed-size tensor.

### 3.3 Robbins–Monro step sizes (learning-rate schedule)

The learning rate is per-(state, action):

$$
\alpha_t(s,a) = \frac{1}{1 + N_t(s,a)}
$$

where $N_t(s,a)$ is how many times we updated that entry (stored in `visit.npy`).

This schedule satisfies the Robbins–Monro conditions used in convergence proofs:

$$
\sum_t \alpha_t(s,a) = \infty
$$

and

$$
\sum_t \alpha_t(s,a)^2 < \infty.
$$

Intuition: early updates are big (fast learning), later updates shrink (stability).

### 3.4 Bounded rewards and bounded Q-values

Rewards are bounded (+15 and −1000).  
I additionally keep Q-values bounded with a safety cap (±10,000). This is a practical safeguard
and matches the “boundedness” assumptions that appear across tabular Q-learning analyses.

### 3.5 Exploration / coverage

Convergence theorems require that relevant state–action pairs are sampled sufficiently often.
I approximate this with ε-greedy exploration during training (and greedy evaluation during watch).

### 3.6 What “converges” means here

In theory (under standard assumptions), tabular Q-learning converges **almost surely**
to the optimal action-value function $Q^*$. In practice, we cannot run forever and the
discretisation introduces approximation error, so what I observe is:

- Q-values stabilising (updates shrink as visits increase)
- a policy that improves on average (smoothed learning curves rise)
- remaining variance because Flappy Bird is stochastic and our state space is coarse

---

## 4) Limitations (honest notes)

- This is **not deep RL**: it does not generalise beyond the bins we define.
- The choice of discretisation (bin sizes) is a major driver of performance.
- With $\gamma = 1$ (no discounting of future rewards), learning can be noisier; dense reward shaping helps.
- Even a “good” policy can have very different outcomes across episodes due to randomness.

---

## 5) References included in this repo

- `FlappyQ.pdf` — Applying Q-Learning to Flappy Bird (Ebeling‑Rump et al.)
- `Regehr_Ayoub_2021_Elementary_Q_learning_convergence.pdf` — An Elementary Proof that Q-learning Converges Almost Surely (Regehr & Ayoub, arXiv:2108.02827)
