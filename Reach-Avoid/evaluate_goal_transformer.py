"""Evaluate GoalTransformerModel on the continuous goal task (same protocol as
evaluate_goal_es2.py).  The transformer has no history channel, so its observation
is the two scans plus the goal vector; an adapter drops the trace slot the
shared rollout provides."""

import argparse

import torch

from evaluate_goal_es2 import run_seed
from model.goal_transformer import GoalTransformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_transformer.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    model = GoalTransformerModel(
        num_features=args.num_features,
        num_actions=args.num_actions,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        sensing_range=args.sensing_range,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    rows = [run_seed(model, args, args.random_seed + i)
            for i in range(args.num_seeds)]
    for k in rows[0]:
        vals = [r[k] for r in rows]
        print(f"{k}: {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
