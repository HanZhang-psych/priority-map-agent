import torch.nn as nn


class GoalMLPModel(nn.Module):
    """Goal-conditioned MLP baseline for the reach-avoid task.

    Input: (batch, 2 * num_features + 2) — two LiDAR scans plus the relative
    goal vector (pixels), all normalized by sensing_range like MLPModel.
    """

    def __init__(
        self,
        num_features=360,
        num_actions=2,
        sensing_range=800.0,
    ):
        super(GoalMLPModel, self).__init__()

        self.sensing_range = sensing_range
        self.num_features = num_features

        self.mlp_action_layers = nn.Sequential(
            nn.Linear(num_features * 2 + 2, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_actions),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input):
        action = self.mlp_action_layers(input / self.sensing_range)

        return action
