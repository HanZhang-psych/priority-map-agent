"""Evaluate GoalEs2Model on the continuous goal task (the evaluation of
record): a persistent world where reaching a goal (or a timeout) spawns the
next one.  Reports goals per minute and collisions per minute, averaged over
seeds.  --num_balls and --speed_multiplier stress densities and speeds never
seen in the demonstrations."""

import argparse
import math
import random

import torch

from model.goal_es2 import GoalEs2Model
from environment import (
    GOAL_RADIUS,
    CollisionTracker,
    lidar_scan,
    make_world,
    sample_goal,
)


def run_seed(model, args, seed):
    random.seed(seed)
    player, balls = make_world(args.num_balls, args.width, args.height)
    for b in balls:
        if b is not player:
            b.dx *= args.speed_multiplier
            b.dy *= args.speed_multiplier
    tracker = CollisionTracker()
    prev = None
    goals = steps = 0

    def step(goal):
        nonlocal prev, steps
        for b in balls:
            if b is not player:
                b.move()
        d, _ = lidar_scan(player, balls, args.width, args.height, args.num_features)
        if prev is None:
            prev = d.copy()
        gdx, gdy = goal[0] - player.x, goal[1] - player.y
        obs = torch.tensor(prev + d + [gdx, gdy],
                           dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(obs)
        fx = min(max(action[0, 0].item(), -args.max_speed), args.max_speed)
        fy = min(max(action[0, 1].item(), -args.max_speed), args.max_speed)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        prev = d.copy()
        steps += 1
        tracker.update(player, balls)

    while steps < args.max_steps:
        goal = sample_goal(player, args.width, args.height,
                           min_dist=args.min_spawn_dist)
        used = 0
        while steps < args.max_steps and used < args.goal_timeout:
            step(goal)
            used += 1
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                goals += 1
                break

    minutes = steps / 50 / 60
    return {
        "goals_per_min": goals / minutes,
        "collisions_per_min": tracker.count / minutes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/goal_es2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument(
        "--speed_multiplier",
        type=float,
        default=1.0,
        help="scale every ball's velocity (training speeds are 1-3 px/step)",
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    model = GoalEs2Model(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    rows = [run_seed(model, args, args.random_seed + i) for i in range(args.num_seeds)]
    for k in rows[0]:
        vals = [r[k] for r in rows]
        print(f"{k}: {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
