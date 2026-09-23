"""One evaluation-sweep condition: --model x --num_balls x --speed_multiplier.
Appends one CSV row (model, balls, speed, seed, goals/min, collisions/min) per seed."""
import argparse
import csv
import os

import torch

from evaluate_goal_es2 import run_seed
from model.goal_es2 import GoalEs2Model
from model.goal_mlp import GoalMLPModel
from model.goal_transformer import GoalTransformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["es2", "mlp", "transformer"], required=True)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--out", type=str, default="results/sweep.csv")
    args = parser.parse_args()

    if args.model == "es2":
        model = GoalEs2Model(num_features=args.num_features)
        model.load_state_dict(torch.load("pretrained/goal_es2.pth", map_location="cpu"))
        model = model
    elif args.model == "mlp":
        model = GoalMLPModel(num_features=args.num_features)
        model.load_state_dict(torch.load("pretrained/goal_mlp.pth", map_location="cpu"))
    else:
        model = GoalTransformerModel(num_features=args.num_features, d_model=16,
                                     nhead=4, num_encoder_layers=2,
                                     dim_feedforward=64, sensing_range=800.0)
        model.load_state_dict(torch.load("pretrained/goal_transformer.pth",
                                         map_location="cpu"))
    model.eval()

    seeds = [args.random_seed + i for i in range(args.num_seeds)]
    rows = [run_seed(model, args, s) for s in seeds]
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "num_balls", "speed_multiplier", "seed",
                        "goals_per_min", "collisions_per_min"])
        for s, r in zip(seeds, rows):
            w.writerow([args.model, args.num_balls, args.speed_multiplier, s,
                        r["goals_per_min"], r["collisions_per_min"]])
    g = sum(r["goals_per_min"] for r in rows) / len(rows)
    c = sum(r["collisions_per_min"] for r in rows) / len(rows)
    print(args.model, args.num_balls, args.speed_multiplier, round(g, 2), round(c, 2))


if __name__ == "__main__":
    main()
