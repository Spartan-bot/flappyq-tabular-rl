"""
env_simple.py

Minimal wrapper around PLE's FlappyBird to expose:
  - reset() -> (xi, yi, zi)
  - step(action) -> (next_state, reward, done)

State = discretised (x distance, y offset, vertical velocity).
Reward shaping = +15 per frame alive, -1000 at crash.
"""

from __future__ import annotations

import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird

# ------------- bins for discretising -------------
X_BINS = np.arange(0, 501, 10)      # 0..500 step 10
Y_BINS = np.arange(-200, 201, 10)   # -200..200 step 10
Z_BINS = np.arange(-15, 16, 3)      # -15..15 step 3

def state_to_index(x: float, y: float, v: float) -> tuple[int, int, int]:
    """Convert continuous (x,y,v) to discrete (xi, yi, zi) bin indices."""
    x = float(np.clip(x, X_BINS[0], X_BINS[-1]))
    y = float(np.clip(y, Y_BINS[0], Y_BINS[-1]))
    v = float(np.clip(v, Z_BINS[0], Z_BINS[-1]))

    xi = int(np.digitize(x, X_BINS) - 1)
    yi = int(np.digitize(y, Y_BINS) - 1)
    zi = int(np.digitize(v, Z_BINS) - 1)
    return xi, yi, zi

class SimpleFlappyEnv:
    """Tiny PLE wrapper with our shaped reward."""
    def __init__(self, headless: bool = True):
        self.game = FlappyBird()
        self.p = PLE(
            self.game,
            fps=30,
            display_screen=not headless,
            reward_values={"positive": 1.0, "tick": 0.0, "loss": 0.0},
        )
        self.p.init()
        self.actions = self.p.getActionSet()  # [NOOP, FLAP]
        self.last_pass = False

    def _get_state_indices(self) -> tuple[int, int, int]:
        gs = self.game.getGameState()
        x = gs["next_pipe_dist_to_player"]
        y = gs["next_pipe_top_y"] - gs["player_y"]
        v = gs["player_vel"]
        return state_to_index(x, y, v)

    def reset(self) -> tuple[int, int, int]:
        self.p.reset_game()
        self.last_pass = False
        return self._get_state_indices()

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        self.last_pass = False
        old_score = self.game.getScore()

        self.p.act(self.actions[action])
        done = self.p.game_over()
        new_score = self.game.getScore()

        if new_score > old_score:
            self.last_pass = True

        reward = -1000.0 if done else 15.0
        return self._get_state_indices(), reward, done
