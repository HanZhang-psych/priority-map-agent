"""Generate expert demonstrations for the continuous goal-directed task.

The potential-field expert (repulsive from LiDAR rays + attractive toward
the visible goal) plays the continuous task: reaching a goal (or a timeout)
spawns the next one in the same persistent world.  Rows include enough
state for the trainer to rebuild the world-space goal trace: current scans,
current goal vector, player position, and goal-spawn events.
"""

import argparse
import csv
import math
import random

from environment import (
    GOAL_RADIUS,
    CollisionTracker,
    expert_goal_action,
    lidar_scan,
    make_world,
    sample_goal,
)


def run_episode(args, writer, episode_id):
    player, balls = make_world(args.num_balls, args.width, args.height)
    tracker = CollisionTracker()
    goals_reached = 0
    step_id = 0

    def write_step(goal, goal_spawn):
        nonlocal step_id
        for ball in balls:
            if ball is not player:
                ball.move()
        distances, _ = lidar_scan(
            player, balls, args.width, args.height, args.num_features
        )
        goal_dx, goal_dy = goal[0] - player.x, goal[1] - player.y

        fx, fy = expert_goal_action(distances, goal_dx, goal_dy, args.num_features)
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))
        tracker.update(player, balls)

        if step_id % args.record_every == 0:
            writer.writerow(
                [fx, fy]
                + distances
                + [
                    goal_dx, goal_dy,
                    player.x, player.y,
                    goal[0], goal[1],
                    1,                  # goal_present: a goal is always visible
                    int(goal_spawn),
                    0,                  # respawn: the world never teleports
                    episode_id,
                ]
            )
        step_id += 1

    while step_id < args.max_steps_per_episode:
        goal = sample_goal(player, args.width, args.height,
                           min_dist=args.min_spawn_dist)
        spawned = True
        for _ in range(args.goal_timeout):
            if step_id >= args.max_steps_per_episode:
                break
            write_step(goal, spawned)
            spawned = False
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                goals_reached += 1
                break

    return goals_reached, tracker.count, step_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps_per_episode", type=int, default=3000)
    parser.add_argument("--record_every", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--min_spawn_dist", type=float, default=250)
    parser.add_argument("--output", type=str, default="dataset/data_goal_unbiased_360.csv")
    args = parser.parse_args()

    random.seed(args.random_seed)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["fx", "fy"]
            + [f"scan_{i}" for i in range(args.num_features)]
            + [
                "goal_dx", "goal_dy",
                "player_x", "player_y", "goal_x", "goal_y",
                "goal_present", "goal_spawn", "respawn", "episode",
            ]
        )
        total_goals = total_collisions = total_steps = 0
        for ep in range(args.num_episodes):
            goals, collisions, steps = run_episode(args, writer, ep)
            total_goals += goals
            total_collisions += collisions
            total_steps += steps
            print(f"Episode {ep}: steps={steps}, goals={goals}, collisions={collisions}")

    minutes = total_steps / 50 / 60
    print(
        f"\nExpert summary: {total_goals} goals, {total_collisions} collisions "
        f"over {total_steps} steps ({total_goals / minutes:.2f} goals/min, "
        f"{total_collisions / minutes:.2f} collisions/min)"
    )
    print(f"Demonstrations written to {args.output}")


if __name__ == "__main__":
    main()
