"""
watch.py

Load a trained Q-table and watch the agent play greedily.
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import pygame
from env_simple import SimpleFlappyEnv
from agent_simple import QLearningAgent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Run directory under results/")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    q_path = os.path.join(args.dir, "q_table.npy")
    if not os.path.exists(q_path):
        print("No q_table.npy found in:", args.dir)
        sys.exit(1)

    env = SimpleFlappyEnv(headless=False)
    agent = QLearningAgent()
    agent.Q = np.load(q_path)

    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    for ep in range(1, args.episodes+1):
        s = env.reset()
        done = False
        frames = 0
        pipes = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)

            xi, yi, zi = s
            a = int(np.argmax(agent.Q[xi, yi, zi, :]))
            s2, r, done = env.step(a)

            if r == 15.0:
                frames += 1
            if env.last_pass:
                pipes += 1

            env.game.screen.blit(
                font.render(f"Ep {ep}/{args.episodes}  Pipes: {pipes}  Frames: {frames}", True, (255,255,255)),
                (5, 5)
            )
            pygame.display.flip()

            s = s2
            if args.fps > 0:
                clock.tick(args.fps)

        print(f"[Agent] Episode {ep}: pipes={pipes} frames={frames}")

    print("Finished. Press Esc or close window to exit.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit(0)
        clock.tick(30)

if __name__ == "__main__":
    main()
