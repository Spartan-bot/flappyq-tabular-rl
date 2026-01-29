"""
train.py

Beginner-friendly training loop:
  - creates env + agent
  - runs episodes
  - logs scores to scores.csv
  - saves q_table.npy and visit.npy in a timestamped results/ folder
"""

from __future__ import annotations
import argparse, os, csv
from datetime import datetime
import numpy as np
from tqdm import trange

from env_simple import SimpleFlappyEnv
from agent_simple import QLearningAgent

def make_run_dir(base="results") -> str:
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = os.path.join(base, stamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--headless", action="store_true", help="Run without opening the game window (faster)")
    args = ap.parse_args()

    run_dir = make_run_dir()
    print(f"Run dir: {run_dir}")

    env = SimpleFlappyEnv(headless=args.headless)
    agent = QLearningAgent()

    csv_path = os.path.join(run_dir, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "frames", "pipes"])  # header

        for ep in trange(args.episodes, desc="Training"):
            eps = agent.epsilon_for_episode(ep, max_ep=args.episodes)

            s = env.reset()
            done = False
            frames = 0
            pipes = 0

            while not done:
                a = agent.act(s, eps)
                s2, r, done = env.step(a)

                if r == 15.0:
                    frames += 1
                if env.last_pass:
                    pipes += 1

                agent.learn(s, a, r, s2, done)
                s = s2

            writer.writerow([ep, frames, pipes])

    np.save(os.path.join(run_dir, "q_table.npy"), agent.Q)
    np.save(os.path.join(run_dir, "visit.npy"), agent.visit)

    print("\n✓ Training finished")
    print("  Q-table: ", os.path.join(run_dir, "q_table.npy"))
    print("  Visits : ", os.path.join(run_dir, "visit.npy"))
    print("  Scores : ", csv_path)

if __name__ == "__main__":
    main()
