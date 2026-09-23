"""The environment: world, dynamics, sensing, goals, and the scripted expert.

Keeps the moving-ball dynamics and 360-ray LiDAR of evaluate.py, and adds a
destination (goal) the ego must reach while avoiding the balls.  Used by
expert_goal.py (demonstration generation), evaluate_goal_es2.py and
evaluate_priming.py (closed-loop evaluation), and the tutorial notebook.
"""

import math
import random

# ---------------- Default parameters (match evaluate.py / expert.py) ----------------
NUM_RAYS = 360
POTENTIAL_FIELD_CONSTANT = 10000  # Repulsive force scale (as in expert.py)
MIN_DISTANCE_FOR_FORCE = 10
MAX_SPEED = 10

# Goal-related defaults
GOAL_RADIUS = 30  # Goal counts as reached within this distance (pixels)
# Wall repulsion summed over all rays is ~764000/r^2 perpendicular to a wall at
# distance r, which balances ATTRACT_FORCE=60 at r~113.  Goals therefore stay
# at least 140 px from walls so the attractive field always wins.
GOAL_WALL_MARGIN = 140  # Goals are sampled at least this far from walls
GOAL_MIN_SPAWN_DIST = 250  # New goals spawn at least this far from the player
ATTRACT_FORCE = 60.0  # Constant-magnitude attraction toward the goal
ATTRACT_TAPER_DIST = 60.0  # Attraction ramps down linearly inside this range


class Ball:
    def __init__(self, x, y, radius, color, speed, width=800, height=800):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        self.width = width
        self.height = height
        # For moving balls, choose a random initial direction.
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def move(self):
        self.prev_x, self.prev_y = self.x, self.y
        self.x += self.dx
        self.y += self.dy

        # Bounce off the walls.
        if self.x - self.radius < 0 or self.x + self.radius > self.width:
            self.dx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > self.height:
            self.dy *= -1

    def collides_with(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance < self.radius + other.radius


def make_world(num_balls, width, height):
    """Create the player and moving balls exactly like evaluate.py."""
    player = Ball(width // 2, height // 2, 20, (255, 0, 0), 0, width, height)
    balls = []
    margin = 10
    min_distance = player.radius + 15 + margin
    for _ in range(num_balls):
        while True:
            x = random.randint(30, width - 30)
            y = random.randint(30, height - 30)
            if math.hypot(x - player.x, y - player.y) >= min_distance:
                break
        balls.append(Ball(x, y, 20, (0, 0, 255), random.randint(1, 3),
                          width, height))
    balls.append(player)
    return player, balls


def lidar_scan(player, balls, width, height, num_rays=NUM_RAYS):
    """360-degree LiDAR scan (identical geometry to evaluate.py)."""
    distances = []
    intersections = []
    for i in range(num_rays):
        rad = math.radians(i * (360 / num_rays))
        dx = math.cos(rad)
        dy = math.sin(rad)

        min_distance_val = float("inf")
        intersection_point = None

        # Walls
        if dx != 0:
            t = (0 - player.x) / dx
            if t > 0:
                y_int = player.y + t * dy
                if 0 <= y_int <= height and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (0, y_int)
            t = (width - player.x) / dx
            if t > 0:
                y_int = player.y + t * dy
                if 0 <= y_int <= height and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (width, y_int)
        if dy != 0:
            t = (0 - player.y) / dy
            if t > 0:
                x_int = player.x + t * dx
                if 0 <= x_int <= width and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (x_int, 0)
            t = (height - player.y) / dy
            if t > 0:
                x_int = player.x + t * dx
                if 0 <= x_int <= width and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (x_int, height)

        # Balls
        for ball in balls:
            if ball is player:
                continue
            bx = ball.x - player.x
            by = ball.y - player.y
            A = dx**2 + dy**2
            B = -2 * (bx * dx + by * dy)
            C = bx**2 + by**2 - ball.radius**2
            discriminant = B**2 - 4 * A * C
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-B - sqrt_disc) / (2 * A)
                t2 = (-B + sqrt_disc) / (2 * A)
                if t1 > 0 and t1 < min_distance_val:
                    min_distance_val = t1
                    intersection_point = (player.x + t1 * dx, player.y + t1 * dy)
                if t2 > 0 and t2 < min_distance_val:
                    min_distance_val = t2
                    intersection_point = (player.x + t2 * dx, player.y + t2 * dy)

        distances.append(min_distance_val)
        intersections.append(intersection_point)
    return distances, intersections


def sample_goal(player, width, height, min_dist=GOAL_MIN_SPAWN_DIST, margin=GOAL_WALL_MARGIN):
    """Sample a goal position away from walls and not too close to the player."""
    while True:
        gx = random.uniform(margin, width - margin)
        gy = random.uniform(margin, height - margin)
        if math.hypot(gx - player.x, gy - player.y) >= min_dist:
            return gx, gy


def expert_goal_action(distances, goal_dx, goal_dy, num_rays=NUM_RAYS):
    """Goal-directed potential-field expert.

    Repulsive term: identical to expert.py (inverse-square from every LiDAR ray).
    Attractive term: constant-magnitude pull toward the goal (uniform-gradient
    potential), tapered linearly near the goal so the ego settles onto it.
    Returns the clipped (fx, fy) applied to the player.
    """
    net_fx = 0.0
    net_fy = 0.0
    for i, d in enumerate(distances):
        effective_d = max(d, MIN_DISTANCE_FOR_FORCE)
        force_magnitude = POTENTIAL_FIELD_CONSTANT / (effective_d**2)
        ray_angle = math.radians(i * (360 / num_rays))
        net_fx += -math.cos(ray_angle) * force_magnitude
        net_fy += -math.sin(ray_angle) * force_magnitude

    goal_dist = math.hypot(goal_dx, goal_dy)
    if goal_dist > 1e-6:
        magnitude = ATTRACT_FORCE * min(1.0, goal_dist / ATTRACT_TAPER_DIST)
        net_fx += magnitude * goal_dx / goal_dist
        net_fy += magnitude * goal_dy / goal_dist

    fx = min(max(net_fx, -MAX_SPEED), MAX_SPEED)
    fy = min(max(net_fy, -MAX_SPEED), MAX_SPEED)
    return fx, fy


class CollisionTracker:
    """Counts collision *events* (contact onsets) so an overlap lasting several
    frames counts once.  A ball must separate before it can register again."""

    def __init__(self):
        self.in_contact = set()
        self.count = 0

    def update(self, player, balls):
        new_events = 0
        for idx, ball in enumerate(balls):
            if ball is player:
                continue
            if player.collides_with(ball):
                if idx not in self.in_contact:
                    self.in_contact.add(idx)
                    self.count += 1
                    new_events += 1
            else:
                self.in_contact.discard(idx)
        return new_events
