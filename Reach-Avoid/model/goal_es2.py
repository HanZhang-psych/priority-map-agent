import math

import torch
import torch.nn as nn

GRID = 8  # the spatial memory is a GRID x GRID leaky map over the arena


class MlpSenseBlock(nn.Module):
    """The obstacle sense of record: a two-channel salience source -
    proximity and looming - combined by a small shared MLP.

    Each LiDAR distance first passes through a sigmoid with learned per-ray
    sensitivity `k` (the psychophysics of closeness: near = strong, far =
    weak, saturating at both ends).  From the previous and current closeness
    two per-ray channels are built:

        proximity = closeness_now
        looming   = (closeness_now - closeness_prev) * closeness_now

    - PROXIMITY is how close something is right now (static standoff).
    - LOOMING is the change in closeness (approach) gated by current
      closeness, so approach only counts where something is already near.

    The obstacle field is a learned nonlinear function of that two-channel
    source, applied per ray with shared weights:

        obstacle(dir) = MLP([proximity, looming])

    This is the salience-side analog of the goal/history `goal_gain` MLP: a
    learned transfer function from sensed physics to signal strength.  It
    supersedes the additive linear combiner (obstacle = g_C * proximity
    + g_L * looming), of which it is the direct nonlinear generalization -
    the MLP can represent that weighted sum as a special case and, beyond
    it, nonlinear interactions between proximity and looming.  On the expert
    demonstrations it fits markedly better (lower cloning MSE) and, at
    matched goal throughput, collides less across the demonstrated and
    stress regimes.

    The input is the full two-scan vector (prev | current) so the dataloader
    and evaluation harness are unchanged.
    """

    def __init__(self, sensing_range: float, num_features: int = 360,
                 hidden: int = 8):
        super().__init__()
        k_init_value = self.inverse_sigmoid(sensing_range, torch.tensor([0.01]))
        self.k = nn.Parameter(torch.full((num_features,), k_init_value.item()))
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def inverse_sigmoid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-1 / x) * torch.log(y / (2 - y))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

    def forward(self, input):
        feature_size = input.size(1) // 2
        sample = input[:, :feature_size]
        sample_next = input[:, feature_size:]

        closeness_prev = self.sigmoid(sample)
        closeness_now = self.sigmoid(sample_next)
        looming = (closeness_now - closeness_prev) * closeness_now

        # per-ray salience source S(dir) = [proximity, looming]; the MLP acts
        # on the last dim (2 -> hidden -> 1), broadcast over batch and rays.
        channels = torch.stack([closeness_now, looming], dim=-1)
        return self.mlp(channels).squeeze(-1)


class GoalEs2Model(nn.Module):
    """Goal-conditioned ES2: obstacle field + goal field + beta_H * history field.

    Network input: previous LiDAR scan | current LiDAR scan | goal dx/dy.

    The obstacle field is the MLP obstacle sense of MlpSenseBlock: a two-
    channel salience source (proximity and looming) mapped to a per-ray
    field by a small shared MLP (obstacle = MLP([proximity, looming])).

    The history field comes from the model's own spatial memory - a leaky
    accumulator over an 8x8 grid of the arena in WORLD coordinates
    (allocentric storage).  At each goal spawn:

        W *= (1 - eta_H);  W[cell(goal)] += eta_H        (update_memory)

    The readout is egocentric and reuses the trained goal machinery: every
    cell acts as a faint goal weighted by its accumulated mass
    (history_field), and one scalar beta_H sets how strongly the whole
    memory pulls (forward).  eta_H = how fast memory updates, W = what is
    remembered, beta_H = how much it biases behavior; a one-back "previous
    goal" memory is the eta_H = 1 degenerate case.

    The memory is rollout state, not a learned weight: it is absent from
    the state dict, reset_memory() clears it, and behavioral cloning never
    touches it - the policy trains on (scans, goal) alone, so beta_H and
    eta_H can be swept or fitted afterwards without retraining the network
    (evaluate_statistical_learning.py fits both from demonstrations).

    Precedence rule of record: a visible goal OVERRIDES history - pass
    history_field to forward() only during goal-free periods.
    """

    def __init__(self, num_features=360, num_actions=2, sensing_range=800.0,
                 eta_H=0.15):
        super().__init__()
        self.num_features = num_features
        self.sensing_range = sensing_range

        self.spatial_sense_block = MlpSenseBlock(
            sensing_range=sensing_range, num_features=num_features
        )
        self.goal_gain = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.beta_H = nn.Parameter(torch.tensor(0.0))
        self.eta_H = eta_H          # leaky rate: plain state, not a weight
        self.memory = None          # set by reset_memory()

        self.sense_action_layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_actions),
        )

        angles = torch.tensor(
            [math.radians(i * (360 / num_features)) for i in range(num_features)]
        )
        self.register_buffer("ray_cos", torch.cos(angles))
        self.register_buffer("ray_sin", torch.sin(angles))
        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ---- the leaky spatial memory ----

    def reset_memory(self, width, height):
        """Clear the grid and anchor it to an arena of the given size."""
        cw, ch = width / GRID, height / GRID
        self._arena = (width, height)
        self._centers = torch.tensor(
            [[(i + 0.5) * cw, (j + 0.5) * ch]
             for j in range(GRID) for i in range(GRID)],
            dtype=torch.float32,
        )
        self.memory = torch.zeros(GRID * GRID)

    def update_memory(self, goal):
        """Leaky accumulation at a goal spawn: decay everything, boost
        the goal's cell."""
        width, height = self._arena
        ci = min(int(goal[0] // (width / GRID)), GRID - 1)
        cj = min(int(goal[1] // (height / GRID)), GRID - 1)
        self.memory = (1 - self.eta_H) * self.memory
        self.memory[cj * GRID + ci] += self.eta_H

    def history_field(self, pos):
        """Egocentric readout of the memory from world position pos:
        each cell is a faint goal weighted by its mass.  Returns
        (1, num_features); forward() scales it by beta_H."""
        vec = self._centers - torch.tensor(pos, dtype=torch.float32)
        with torch.no_grad():
            G = self.geometric_field(vec, self.goal_gain)
        return (self.memory.unsqueeze(1) * G).sum(0, keepdim=True)

    # ---- the fields ----

    def geometric_field(self, vec, gain_net):
        """Write relative position vectors into geometric ray-wise fields."""
        dist = torch.norm(vec, dim=1, keepdim=True)
        unit = vec / (dist + 1e-6)
        alignment = unit[:, 0:1] * self.ray_cos + unit[:, 1:2] * self.ray_sin
        gain = gain_net(dist / self.sensing_range)
        return gain * alignment

    def compute_fields(self, input):
        scan_input = input[:, : 2 * self.num_features]
        goal = input[:, 2 * self.num_features :]

        obstacle_field = self.spatial_sense_block(scan_input)
        goal_field = self.geometric_field(goal, self.goal_gain)
        return obstacle_field, goal_field

    def forward(self, input, history_field=None):
        obstacle_field, goal_field = self.compute_fields(input)
        field = obstacle_field + goal_field
        if history_field is not None:
            field = field + self.beta_H * history_field
        return self.sense_action_layers(field)
