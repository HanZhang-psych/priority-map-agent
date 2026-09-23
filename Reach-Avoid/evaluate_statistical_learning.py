"""Statistical learning of likely goal locations: leaky world-grid history.

The history memory is a leaky accumulator over an 8x8 grid of the arena
in WORLD coordinates (allocentric storage): at each goal spawn every
weight decays by (1 - eta) and the goal's cell gains eta.  The readout
is egocentric and reuses the trained goal machinery - each cell is a
faint goal painted through geometric_field(center - player, goal_gain),
weighted by its accumulated mass:

    history_field = beta_H * sum_c W[c] * geometric_field(center_c - player,
                                                          goal_gain)

The priority-field formula is unchanged (obstacle + goal + beta_H *
history), the precedence rule of record holds (a visible goal overrides
history: the history field feeds the map only during goal-free
anticipation windows), and the one-back trace of evaluate_priming.py is
the eta = 1 degenerate case.

Trials are fixation-start (teleport to center; a goal-free anticipation
window; then the goal), with goals at eccentricity 250-330: 70% in a
90-degree wedge centered due-left, the rest in the mirror wedge
due-right (fixed, symmetric, distance-matched regions).

Three stages (the full chain; run any subset via --stages):
  teacher  - the trained checkpoint + grid at ground-truth (--beta_star,
             --eta_star) plays the biased task; goal-free rows recorded.
  fit      - a fresh frozen copy of the checkpoint; ONLY (beta_H, eta)
             are trained, by MSE on the teacher's goal-free rows, with
             the grid replayed differentiably under the candidate eta.
             Teacher and student share the network, so recovery is
             exact-units.
  deploy   - the fitted student in a biased-then-unbiased world (bias
             for the first --switch trials, 50/50 after): the drift
             bias emerges from experienced goals and extinguishes with
             time constant ~1/eta once the bias is removed.
"""

import argparse
import csv
import math
import os
import random

import torch

from model.goal_es2 import GRID, GoalEs2Model
from environment import GOAL_RADIUS, lidar_scan, make_world


def grid_centers(width, height):
    cw, ch = width / GRID, height / GRID
    return torch.tensor([[(i + 0.5) * cw, (j + 0.5) * ch]
                         for j in range(GRID) for i in range(GRID)],
                        dtype=torch.float32)


def cell_of(g, width, height):
    return (min(int(g[1] // (height / GRID)), GRID - 1) * GRID
            + min(int(g[0] // (width / GRID)), GRID - 1))


def sample_goal_trial(rng, biased, width, height,
                      ecc_min=250.0, ecc_max=330.0, p_high=0.7):
    """70% (or 50% when unbiased) in the left wedge (135-225 deg), the
    rest in the mirror right wedge - fixed symmetric distance-matched
    regions."""
    ecc = rng.uniform(ecc_min, ecc_max)
    if rng.random() < (p_high if biased else 0.5):
        ang = math.pi + rng.uniform(-math.pi / 4, math.pi / 4)
    else:
        ang = rng.uniform(-math.pi / 4, math.pi / 4)
    g = (width / 2 + ecc * math.cos(ang), height / 2 + ecc * math.sin(ang))
    return (min(max(g[0], 60), width - 60), min(max(g[1], 60), height - 60))


def play(model, beta, eta, seed, n_trials, biased_until, args,
         record_rows=False):
    """The checkpoint + grid agent (gated) on the fixation-start task.
    Returns (goal-free rows, goal-cell sequence, per-trial anticipatory
    offset: the signed horizontal position at goal onset relative to
    center, in px; negative = leftward)."""
    rng = random.Random(seed)
    random.seed(seed)
    width, height = args.width, args.height
    player, balls = make_world(args.num_balls, width, height)
    model.eta_H = eta
    model.reset_memory(width, height)
    prev_scan = None
    rows, goal_seq = [], []
    drifts = [float("nan")] * n_trials

    def act(goal_vec, hfield):
        nonlocal prev_scan
        d, _ = lidar_scan(player, balls, width, height, args.num_features)
        if prev_scan is None:
            prev_scan = d
        obs = torch.tensor(prev_scan + d + list(goal_vec),
                           dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            of, gf = model.compute_fields(obs)
            f = of + gf
            if hfield is not None:
                f = f + beta * hfield
            a = model.sense_action_layers(f)[0]
        scan = prev_scan
        prev_scan = d
        return scan, d, (min(max(a[0].item(), -args.max_speed), args.max_speed),
                         min(max(a[1].item(), -args.max_speed), args.max_speed))

    def step_world():
        for b in balls:
            if b is not player:
                b.move()

    def move(fx, fy):
        player.x = max(player.radius, min(width - player.radius,
                                          player.x + int(fx)))
        player.y = max(player.radius, min(height - player.radius,
                                          player.y + int(fy)))

    for trial in range(n_trials):
        player.x, player.y = width // 2, height // 2
        prev_scan = None
        if trial > 0:
            x0 = player.x
            for t in range(args.goal_free_steps):
                step_world()
                hf = model.history_field((player.x, player.y))
                px, py = player.x, player.y
                pscan, d, (fx, fy) = act((0.0, 0.0), hf)
                if record_rows and t % args.record_every == 0:
                    rows.append((trial, list(pscan), list(d), px, py, fx, fy))
                move(fx, fy)
            drifts[trial] = player.x - x0  # signed horizontal offset at goal
            #                                onset, relative to center (px);
            #                                negative = leftward
        goal = sample_goal_trial(rng, trial < biased_until, width, height,
                                 args.ecc_min, args.ecc_max, args.p_high)
        for t in range(args.goal_timeout):
            step_world()
            _, _, (fx, fy) = act((goal[0] - player.x, goal[1] - player.y),
                                 None)
            move(fx, fy)
            if math.hypot(goal[0] - player.x, goal[1] - player.y) < GOAL_RADIUS:
                break
        goal_seq.append(cell_of(goal, width, height))
        model.update_memory(goal)
    return rows, goal_seq, drifts


def fit_history_params(model, rows, goal_seq, args, epochs=400, verbose=True):
    """Fit ONLY (beta_H, eta) on goal-free rows, all network weights
    frozen; the grid is replayed differentiably under the candidate eta
    (the agent-side analog of the search model's history fitting)."""
    centers = grid_centers(args.width, args.height)
    n = len(rows)
    trials_t = torch.tensor([r[0] for r in rows])
    scans = torch.tensor([r[1] + r[2] for r in rows], dtype=torch.float32)
    pos = torch.tensor([[r[3], r[4]] for r in rows], dtype=torch.float32)
    target = torch.tensor([[r[5], r[6]] for r in rows], dtype=torch.float32)
    with torch.no_grad():
        obs_field = model.spatial_sense_block(scans)
        vec = centers.unsqueeze(0) - pos.unsqueeze(1)
        G = model.geometric_field(vec.reshape(-1, 2),
                                  model.goal_gain).reshape(n, GRID * GRID, -1)
    onehots = torch.zeros(len(goal_seq), GRID * GRID)
    for t, c in enumerate(goal_seq):
        onehots[t, c] = 1.0

    raw_beta = torch.tensor(-2.0, requires_grad=True)
    raw_eta = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([raw_beta, raw_eta], lr=0.05)
    for epoch in range(epochs):
        eta = torch.sigmoid(raw_eta)
        beta = torch.nn.functional.softplus(raw_beta)
        W = torch.zeros(GRID * GRID)
        Ws = []
        for t in range(len(goal_seq)):
            Ws.append(W)
            W = (1 - eta) * W + eta * onehots[t]
        hf = torch.einsum("nc,ncr->nr", torch.stack(Ws)[trials_t], G)
        pred = model.sense_action_layers(obs_field + beta * hf)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch}: loss {loss.item():.5f} "
                  f"eta {eta.item():.3f} beta {beta.item():.3f}", flush=True)
    return (torch.nn.functional.softplus(raw_beta).item(),
            torch.sigmoid(raw_eta).item(), loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pretrained/goal_es2.pth")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--goal_timeout", type=int, default=300)
    parser.add_argument("--goal_free_steps", type=int, default=50)
    parser.add_argument("--record_every", type=int, default=5)
    parser.add_argument("--ecc_min", type=float, default=250.0)
    parser.add_argument("--ecc_max", type=float, default=330.0)
    parser.add_argument("--p_high", type=float, default=0.7)
    parser.add_argument("--beta_star", type=float, default=0.2)
    parser.add_argument("--eta_star", type=float, default=0.15)
    parser.add_argument("--demo_trials", type=int, default=200)
    parser.add_argument("--deploy_trials", type=int, default=180)
    parser.add_argument("--switch", type=int, default=90)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=100,
                        help="first deploy seed; seeds are seed_start..+num_seeds-1")
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results_compare",
                        help="dir for statlearn_drift.csv / statlearn_params.csv")
    args = parser.parse_args()

    model = GoalEs2Model(num_features=args.num_features)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    assert model.beta_H.item() == 0.0, "base policy must be history-disabled"

    print(f"1. TEACHER (beta* = {args.beta_star}, eta* = {args.eta_star}): "
          f"{args.demo_trials} biased trials...", flush=True)
    rows, goal_seq, tdrifts = play(model, args.beta_star, args.eta_star,
                                   args.random_seed, args.demo_trials,
                                   args.demo_trials, args, record_rows=True)
    td = [d for d in tdrifts if not math.isnan(d)]
    teacher_offset = sum(td) / len(td)
    print(f"   {len(rows)} goal-free rows; teacher anticipatory offset "
          f"{teacher_offset:+.1f} px (negative = left)", flush=True)

    print("2. FIT (beta_H, eta), policy frozen...", flush=True)
    beta_hat, eta_hat, mse = fit_history_params(model, rows, goal_seq, args)
    print(f"   RECOVERY: beta* {args.beta_star} -> {beta_hat:.3f}   "
          f"eta* {args.eta_star} -> {eta_hat:.3f}   (MSE {mse:.5f})",
          flush=True)

    print(f"3. DEPLOY fitted student and the beta_H = 0 control, "
          f"{args.num_seeds} seeds each: bias for {args.switch} trials, "
          f"then 50/50...", flush=True)

    def run_arm(beta):
        curves = []
        for i in range(args.num_seeds):
            _, _, drifts = play(model, beta, eta_hat, args.seed_start + i,
                                args.deploy_trials, args.switch, args)
            first = [d for t, d in enumerate(drifts)
                     if t < args.switch and not math.isnan(d)]
            second = [d for t, d in enumerate(drifts)
                      if t >= args.switch and not math.isnan(d)]
            print(f"   beta={beta:.3f} seed {args.seed_start + i}: biased-half "
                  f"{sum(first)/len(first):+.1f}  unbiased-half "
                  f"{sum(second)/len(second):+.1f} px", flush=True)
            curves.append(drifts)
        return curves

    fitted = run_arm(beta_hat)
    control = run_arm(0.0)

    # Write the results the notebook loads (it no longer runs this chain).
    os.makedirs(args.out_dir, exist_ok=True)
    drift_path = os.path.join(args.out_dir, "statlearn_drift.csv")
    with open(drift_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed", "trial", "offset"])
        for arm, curves in [("fitted", fitted), ("control", control)]:
            for i, drifts in enumerate(curves):
                for t, d in enumerate(drifts):
                    if not math.isnan(d):
                        w.writerow([arm, args.seed_start + i, t, f"{d:.5f}"])
    params_path = os.path.join(args.out_dir, "statlearn_params.csv")
    with open(params_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beta_star", "eta_star", "beta_hat", "eta_hat", "fit_mse",
                    "teacher_offset", "switch", "deploy_trials", "num_seeds",
                    "block"])
        w.writerow([args.beta_star, args.eta_star, f"{beta_hat:.5f}",
                    f"{eta_hat:.5f}", f"{mse:.6f}", f"{teacher_offset:.5f}",
                    args.switch, args.deploy_trials, args.num_seeds, args.block])
    print(f"   wrote {drift_path} and {params_path}", flush=True)


if __name__ == "__main__":
    main()
