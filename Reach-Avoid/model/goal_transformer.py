import math

import torch
import torch.nn as nn


class GoalTransformerModel(nn.Module):
    """Goal-conditioned Transformer baseline for the reach-avoid task.

    Same architecture as TransformerModel, with the two relative-goal
    components appended as two extra tokens (sequence length
    2 * num_features + 2), sharing the sinusoidal positional encoding.
    """

    def __init__(
        self,
        num_features=360,
        num_actions=2,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        sensing_range=10.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.sensing_range = sensing_range
        self.seq_len = num_features * 2 + 2

        self.embedding_projection = nn.Linear(1, d_model)

        self.register_buffer(
            "positional_encoding", self._generate_positional_encoding()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.action_proj = nn.Sequential(
            nn.Linear(self.seq_len, num_actions),
        )

        self._initialize_parameters()

    def _generate_positional_encoding(self):
        encoding = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding  # shape: [seq_len, d_model]

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_features * 2 + 2]
        """
        x = x / self.sensing_range  # Normalize scans and goal alike

        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embedding_projection(x)  # [batch, seq_len, d_model]

        x = x + self.positional_encoding.unsqueeze(0)

        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        x = x.mean(dim=2)  # [batch, seq_len]

        action = self.action_proj(x)  # [batch, num_actions]

        return action
