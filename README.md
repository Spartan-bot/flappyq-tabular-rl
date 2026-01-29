# Flappy Q-Learning — Beginner Tutorial 

A minimal, beginner-friendly project that teaches **tabular Q-learning** using
the PyGame Learning Environment (PLE) version of **Flappy Bird**. This project was built as a learning exercise to connect practical tabular Q-learning with modern convergence theory.

Key features:
- discretising game state (distance to next pipe, vertical offset, velocity)
- using a simple **Q-table** with **ε-greedy** exploration
- applying dense **reward shaping** (+15 per frame alive, −1000 on crash)
- training, plotting a learning curve, and watching the trained agent fly

---

## 1) Setup (macOS + VS Code)

```bash
# 1. Create project folder and open it in VS Code
cd flappy_ql_tutorial_simple

# 2. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install PLE (Flappy Bird environment)
git clone https://github.com/ntasfi/PyGame-Learning-Environment ple-src
pip install -e ple-src
```

Sanity check:
```bash
python -c "from ple.games.flappybird import FlappyBird; g=FlappyBird(); print('Bird colours:', list(g.images['player'].keys()))"
```

---

## 2) Train the agent

```bash
# Headless (faster)
python train.py --episodes 30000 --headless

# Or watch learning (slower, but fun)
python train.py --episodes 5000
```

This creates a timestamped run directory under `results/` with:
- `q_table.npy`  — learned Q-values
- `visit.npy`    — state-action visit counts
- `scores.csv`   — episode metrics (frames & pipes)
- `learning_curve.png` — produced by `plot.py`
- `visit_heatmap.png`  — produced by `plot.py`

---

## 3) Plot results

```bash
python plot.py --dir results/<your-run-folder>
```

This writes `learning_curve.png` and `visit_heatmap.png` into that run folder.

---

## 4) Watch the trained agent

```bash
python watch.py --dir results/<your-run-folder> --episodes 3 --fps 30
```

Close the window or press `Esc` to quit.

---

## 5) What can someone learn from this project

- **State discretisation**: turning continuous Flappy Bird state into integer bins.
- **Tabular Q-learning**: the update rule
  \[Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))\]
- **Exploration vs exploitation** with ε-greedy.
- **Reward shaping** to make learning dense and stable.
- Reading learning curves & visit heat-maps.

---

## 6) Troubleshooting

- If the Pygame window is unresponsive on macOS, click the window once to focus it.
- If imports fail, ensure the venv is activated (`source .venv/bin/activate`).
- If `ple` import fails, re-run `pip install -e ple-src` from this folder.
- If training is very slow, use `--headless` and lower `--episodes` for a quick smoke test.

---

## 7) Credits & Paper Context

This project is a didactic re-implementation inspired by the paper
Flappy Bird Q-learning (Bilkent University EEE546, “Applying Q-Learning
to Flappy Bird”) and the original **FlapPyBird** clone that PLE references.

See `CONTRIBUTIONS.md` for what I have changed, why, and how my setup differs from
the handout (e.g., PLE environment, discretisation choices, plotting).
