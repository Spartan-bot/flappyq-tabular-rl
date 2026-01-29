"""
agent_simple.py

A minimal tabular Q-learning agent:
  - Q table over (xi, yi, zi, action)
  - visit counts for per-(s,a) step size alpha = 1/(1+visits)
  - epsilon-greedy action selection
"""

from __future__ import annotations
import numpy as np
from env_simple import X_BINS, Y_BINS, Z_BINS

N_ACTIONS = 2
GAMMA = 1.0
Q_CAP = 10_000.0

class QLearningAgent:
    def __init__(self):
        shape = (len(X_BINS), len(Y_BINS), len(Z_BINS), N_ACTIONS)
        self.Q = np.zeros(shape, dtype=np.float32)
        self.visit = np.zeros(shape, dtype=np.int64)

    @staticmethod
    def epsilon_for_episode(ep: int, max_ep: int, eps_start: float = 0.10, eps_end: float = 0.02) -> float:
        if ep >= max_ep:
            return eps_end
        frac = ep / float(max_ep)
        return eps_start + frac * (eps_end - eps_start)

    def act(self, state: tuple[int,int,int], eps: float) -> int:
        xi, yi, zi = state
        if np.random.rand() < eps:
            return int(np.random.randint(N_ACTIONS))
        q_vals = self.Q[xi, yi, zi, :]
        return int(np.argmax(q_vals))

    def learn(self, s, a, r, s2, done):
        xi, yi, zi = s
        nxi, nyi, nzi = s2

        self.visit[xi, yi, zi, a] += 1
        n_visits = self.visit[xi, yi, zi, a]
        alpha = 1.0 / float(1 + n_visits)

        old_q = float(self.Q[xi, yi, zi, a])

        if done:
            target = r
        else:
            best_next = float(np.max(self.Q[nxi, nyi, nzi, :]))
            target = r + GAMMA * best_next

        new_q = old_q + alpha * (target - old_q)

        if new_q > Q_CAP: new_q = Q_CAP
        if new_q < -Q_CAP: new_q = -Q_CAP

        self.Q[xi, yi, zi, a] = new_q
