# priority-map-agent

Code for *A Priority Field for Movement Control in Artificial Agents*.

A priority map built from salience, goal and history is fitted to human
first fixations in visual search. The same computation is then extended
to a priority field over movement directions that controls an agent in a
2-D reach–avoid task.

## Layout

| Folder | Paper section | Contents |
| --- | --- | --- |
| [`Visual-Search/`](Visual-Search/) | A priority map for visual search (Fig. 2) | Data pooling, display reconstruction, the 4-parameter model and its analyses (`tutorial_visual_search.ipynb`) |
| [`Reach-Avoid/`](Reach-Avoid/) | A priority field for agent movements (Figs. 3–5) | Arena, scripted expert, priority-field / MLP / transformer agents, training, evaluation sweeps, statistical learning, and the analysis notebook (`tutorial_action_agent.ipynb`) |

`Reach-Avoid/` includes the expert demonstrations, the three trained agents
(`pretrained/goal_*.pth`), the per-seed results behind Figs. 4–5
(`results_compare/`) and the demo clips (`demo_videos/`). The notebook can
redraw the figures without retraining.

The human saccade data are **not** included. They come from Drennan &
Gaspelin (OSF: https://osf.io/q27ph/). See [`Visual-Search/README.md`](Visual-Search/README.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`render_demo.py` also needs `ffmpeg` on the PATH.

## Reproduce

### Visual search

```bash
cd Visual-Search
python pool_data.py --data_dir "<OSF download>/Data Files" --out_dir dataset
python build_contexts.py
jupyter lab tutorial_visual_search.ipynb
```

### Reach–avoid agent

Run from `Reach-Avoid/`:

```bash
# Expert demonstrations (20 episodes, 10 obstacles)
python expert_goal.py

# Behavioral cloning, 500 epochs (also train_goal_mlp.py / train_goal_transformer.py)
python train_goal_es2.py --num_epochs 500 --log_path results_compare/loss_goal_es2.csv

# Evaluation sweep: one condition per call, 20 seeds (Fig. 4)
python eval_sweep.py --model es2 --num_balls 50 --speed_multiplier 1.0 --num_seeds 20 --out results_compare/sweep_es2.csv

# Statistical learning with the history field (Fig. 5)
python evaluate_statistical_learning.py --num_seeds 20

# Demo clips (Fig. 3)
python render_demo.py --agent es2 --num_balls 50 --out demo_videos/es2_crowded.mp4

jupyter lab tutorial_action_agent.ipynb
```

## License

PolyForm Noncommercial 1.0.0. See [`LICENSE.txt`](LICENSE.txt).
