# Flappy Q-Learning — Beginner Tutorial

A minimal, beginner-friendly project that teaches **tabular Q-learning** using
the PyGame Learning Environment (PLE) version of **Flappy Bird**.

Key features:
- discretising game state (distance to next pipe, vertical offset, velocity)
- using a simple **Q-table** with **ε-greedy** exploration
- applying dense **reward shaping** (+15 per frame alive, −1000 on crash)
- training, plotting a learning curve, and watching the trained agent fly

---

## 1) Setup (macOS)

```bash
cd flappyq-tabular-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/ntasfi/PyGame-Learning-Environment ple-src
pip install -e ple-src
```

---

## 2) What can someone learn from this project

The core tabular Q-learning update rule is:

$$
Q(s,a) \leftarrow Q(s,a)
+ \alpha \bigl(
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\bigr)
$$

This illustrates how action values are updated toward observed reward plus the
best estimated future return.
