"""Headless arena demo videos (field-colored LiDAR rays; Fig. 3 and supplementary clips).

Modes:
  reachavoid  - continuous reach-avoid; --num_balls / --speed set the scenario.
  statlearn   - fixation-start biased task (grid memory); shows goalless drift.

The field agent (es2) colors each LiDAR ray by the INTEGRATED priority field
(red = push/avoid -> green = pull/approach). MLP/transformer have no field, so
their clips show the arena + agent + movement only.

Frames are piped to ffmpeg (rawvideo -> h264 mp4). Run with the escaping_ball env.
"""
import argparse, colorsys, math, os, random, subprocess
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
import torch

from environment import GOAL_RADIUS, CollisionTracker, lidar_scan, make_world, sample_goal
from model.goal_es2 import GoalEs2Model
from model.goal_mlp import GoalMLPModel
from model.goal_transformer import GoalTransformerModel

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

W = H = 800
HEADER = 40                     # white caption band above the arena
CANVAS_H = H + HEADER
BLUE, RED, GREEN, PURPLE, GRAY = (60,90,210),(220,30,30),(17,136,68),(120,60,160),(200,200,200)

def load_agent(agent):
    if agent == "es2":
        m = GoalEs2Model(num_features=360)
        m.load_state_dict(torch.load("pretrained/goal_es2.pth", map_location="cpu", weights_only=True))
    elif agent == "mlp":
        m = GoalMLPModel(num_features=360)
        m.load_state_dict(torch.load("pretrained/goal_mlp.pth", map_location="cpu", weights_only=True))
    else:
        m = GoalTransformerModel(num_features=360, d_model=16, nhead=4,
                                 num_encoder_layers=2, dim_feedforward=64, sensing_range=800.0)
        m.load_state_dict(torch.load("pretrained/goal_transformer.pth", map_location="cpu", weights_only=True))
    m.eval()
    return m

def draw_arrow(s, color, p0, p1, wdt=4, head=12):
    pygame.draw.line(s, color, p0, p1, wdt)
    a = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    l = (p1[0]-head*math.cos(a-0.4), p1[1]-head*math.sin(a-0.4))
    r = (p1[0]-head*math.cos(a+0.4), p1[1]-head*math.sin(a+0.4))
    pygame.draw.polygon(s, color, [p1, l, r])

def ffmpeg(path, fps):
    return subprocess.Popen(
        ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
         "-video_size",f"{W}x{CANVAS_H}","-framerate",str(fps),"-i","-",
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",path], stdin=subprocess.PIPE)

def push(proc, screen):
    frame = pygame.surfarray.array3d(screen).transpose(1,0,2)  # (H,W,3) RGB
    proc.stdin.write(frame.astype(np.uint8).tobytes())

def draw_scene(screen, font, balls, player, field, inter, goal, prev_goal, fx, fy, hud):
    arena = pygame.Surface((W, H))          # 800x800 scene, drawn in its own coords
    arena.fill((255,255,255))
    for b in balls:
        if b is not player:
            pygame.draw.circle(arena, BLUE, (int(b.x),int(b.y)), b.radius)
    if field is not None:                       # field agent: priority-colored rays
        vmax = max(float(abs(field).max()), 1e-6)
        for i, pt in enumerate(inter):
            if pt:
                a = math.radians(i)
                start = (player.x+player.radius*math.cos(a), player.y+player.radius*math.sin(a))
                n = float(field[i])/vmax
                r,g,bl = colorsys.hsv_to_rgb(0.165*(1-n), 1, 1)
                pygame.draw.line(arena, (int(r*255),int(g*255),int(bl*255)), start, pt, 1)
    if goal is not None:
        pygame.draw.circle(arena, GREEN, (int(goal[0]),int(goal[1])), GOAL_RADIUS, 3)
    if prev_goal is not None:
        pygame.draw.circle(arena, PURPLE, (int(prev_goal[0]),int(prev_goal[1])), 7)
    pygame.draw.circle(arena, RED, (int(player.x),int(player.y)), player.radius)
    if math.hypot(fx,fy) > 0:
        draw_arrow(arena, RED, (player.x,player.y), (player.x+fx*6, player.y+fy*6))
    screen.fill((255,255,255))                  # caption band + arena below it
    screen.blit(font.render(hud, True, (0,0,0)), (12, 9))
    screen.blit(arena, (0, HEADER))

def clamp(v): return max(-10, min(10, v))

def smooth_field(field, sigma):
    """Circular Gaussian smoothing over the 360 directions, for DISPLAY only."""
    if not sigma or gaussian_filter1d is None:
        return field
    return gaussian_filter1d(field, sigma=sigma, mode="wrap")


def run_reachavoid(agent, model, num_balls, speed, steps, out, fps, seed=42, smooth=6):
    random.seed(seed)
    player, balls = make_world(num_balls, W, H)
    for b in balls:
        if b is not player:
            b.dx *= speed; b.dy *= speed
    tracker = CollisionTracker(); prev=None; goals=0; used=0
    goal = clear_reach_goal(player)
    pygame.init(); screen = pygame.display.set_mode((W,CANVAS_H)); font = pygame.font.SysFont("Arial",22)
    proc = ffmpeg(out, fps)
    for t in range(steps):
        for b in balls:
            if b is not player: b.move()
        d, inter = lidar_scan(player, balls, W, H, 360)
        if prev is None: prev = d.copy()
        obs = torch.tensor(prev + d + [goal[0]-player.x, goal[1]-player.y], dtype=torch.float32).unsqueeze(0)
        field = None
        with torch.no_grad():
            if agent == "es2":
                of, gf = model.compute_fields(obs); field = (of+gf)[0].numpy()
                action = model.sense_action_layers(torch.tensor(field, dtype=torch.float32).unsqueeze(0))
                field = smooth_field(field, smooth)   # display-only smoothing
            else:
                action = model(obs)
        fx, fy = clamp(action[0,0].item()), clamp(action[0,1].item())
        player.x = max(player.radius, min(W-player.radius, player.x+int(fx)))
        player.y = max(player.radius, min(H-player.radius, player.y+int(fy)))
        prev = d.copy(); used += 1; tracker.update(player, balls)
        hud = f"{AGENT_LABEL[agent]}   {num_balls} balls, {speed:g}x speed   goals {goals}   collisions {tracker.count}"
        draw_scene(screen, font, balls, player, field, inter, goal, None, fx, fy, hud)
        push(proc, screen)
        if used >= 300 or math.hypot(goal[0]-player.x, goal[1]-player.y) < GOAL_RADIUS:
            if math.hypot(goal[0]-player.x, goal[1]-player.y) < GOAL_RADIUS: goals += 1
            goal = clear_reach_goal(player); used = 0
    proc.stdin.close(); proc.wait(); pygame.quit()
    print(f"wrote {out}  ({steps} frames, goals {goals}, collisions {tracker.count})")

AGENT_LABEL = {"es2":"Priority-Field Agent","mlp":"MLP Agent","transformer":"Transformer Agent"}


WALL_MARGIN = 120   # demo only: keep goals clear of walls so the boundary's
#                     repulsion never opposes the goal (a potential-field local
#                     minimum). Does not touch the eval/training goal sampling.

def _clear(gx, gy, width, height, margin=WALL_MARGIN):
    return margin <= gx <= width - margin and margin <= gy <= height - margin

def biased_goal(rng, width, height, ecc=(250.,330.), p_high=0.7):
    """70% in a 90-deg wedge due-left, else the mirror wedge due-right;
    rejection-resampled to stay >= WALL_MARGIN px from every wall."""
    for _ in range(100):
        e = rng.uniform(*ecc)
        ang = (math.pi if rng.random() < p_high else 0.0) + rng.uniform(-math.pi/4, math.pi/4)
        gx, gy = width/2 + e*math.cos(ang), height/2 + e*math.sin(ang)
        if _clear(gx, gy, width, height):
            return (gx, gy)
    return (min(max(gx,WALL_MARGIN),width-WALL_MARGIN), min(max(gy,WALL_MARGIN),height-WALL_MARGIN))

def clear_reach_goal(player):
    """reach-avoid demo goal, kept clear of walls (same rationale)."""
    for _ in range(100):
        g = sample_goal(player, W, H, min_dist=250)
        if _clear(g[0], g[1], W, H):
            return g
    return g


def run_statlearn(model, steps, out, fps, seed=1, smooth=12, beta=0.2, eta=0.15,
                  warmup=12, free=50, num_balls=10):
    """Field agent on the fixation-start biased task with grid memory. During
    each goal-free window the map is obstacle + beta*history and the agent
    drifts toward the frequently rewarded (left) region -- the goalless drift.
    A few silent warm-up goals pre-build the bias so the drift is visible."""
    rng = random.Random(seed); random.seed(seed)
    model.eta_H = eta; model.reset_memory(W, H)
    player, balls = make_world(num_balls, W, H)
    for _ in range(warmup):                     # establish the bias silently
        model.update_memory(biased_goal(rng, W, H))
    tracker = CollisionTracker()
    pygame.init(); screen = pygame.display.set_mode((W, CANVAS_H)); font = pygame.font.SysFont("Arial",18)
    proc = ffmpeg(out, fps); done = 0

    def step_and_draw(goal_vec, hf, goal_pt, phase):
        nonlocal done
        for b in balls:
            if b is not player: b.move()
        d, inter = lidar_scan(player, balls, W, H, 360)
        if step_and_draw.prev is None: step_and_draw.prev = d.copy()
        obs = torch.tensor(step_and_draw.prev + d + list(goal_vec), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            of, gf = model.compute_fields(obs)
            field = of + gf + (beta * hf if hf is not None else 0.0)
            field = field[0].numpy()
            action = model.sense_action_layers(torch.tensor(field, dtype=torch.float32).unsqueeze(0))
        fx, fy = clamp(action[0,0].item()), clamp(action[0,1].item())
        player.x = max(player.radius, min(W-player.radius, player.x+int(fx)))
        player.y = max(player.radius, min(H-player.radius, player.y+int(fy)))
        step_and_draw.prev = d.copy(); tracker.update(player, balls)
        hud = f"Statistical learning   {phase}"
        draw_scene(screen, font, balls, player, smooth_field(field, smooth), inter,
                   goal_pt, None, fx, fy, hud)
        push(proc, screen); done += 1

    while done < steps:
        player.x, player.y = W//2, H//2; step_and_draw.prev = None   # fixation start
        for _ in range(free):                    # goal-free window: history drives the map
            if done >= steps: break
            hf = model.history_field((player.x, player.y))
            step_and_draw((0.0,0.0), hf, None, "ANTICIPATION: goalless drift toward the frequent region")
        goal = biased_goal(rng, W, H)            # goal appears, history gated off
        used = 0
        while done < steps and used < 300:
            step_and_draw((goal[0]-player.x, goal[1]-player.y), None, goal, "GOAL VISIBLE")
            used += 1
            if math.hypot(goal[0]-player.x, goal[1]-player.y) < GOAL_RADIUS: break
        model.update_memory(goal)
    proc.stdin.close(); proc.wait(); pygame.quit()
    print(f"wrote {out}  ({done} frames, statlearn)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["es2","mlp","transformer"], default="es2")
    ap.add_argument("--num_balls", type=int, default=50)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--smooth", type=float, default=6,
                    help="circular Gaussian sigma (deg) for DISPLAY field smoothing; 0 = off")
    ap.add_argument("--mode", choices=["reachavoid","statlearn"], default="reachavoid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "statlearn":
        run_statlearn(load_agent("es2"), a.steps, a.out, a.fps, smooth=a.smooth)
    else:
        run_reachavoid(a.agent, load_agent(a.agent), a.num_balls, a.speed, a.steps, a.out, a.fps,
                       smooth=a.smooth)
