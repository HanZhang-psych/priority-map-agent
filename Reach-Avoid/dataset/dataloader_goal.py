import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class GoalSenseDataset(Dataset):
    """
    Dataset for goal-directed reach-avoid demonstrations.
    Each sample concatenates:
      - the previous frame's LiDAR scan
      - the current frame's LiDAR scan
      - the current frame's relative goal vector (goal_dx, goal_dy)
    The target is the current frame's (fx, fy).
    Pairs that cross an episode boundary are excluded.
    """

    def __init__(self, data, scan_columns, goal_columns, target_columns, device):
        data = data.reset_index(drop=True)

        # Valid indices: consecutive frame pairs within the same episode.
        episodes = data["episode"].values
        pairs = [i for i in range(len(data) - 1) if episodes[i] == episodes[i + 1]]
        self.pairs = torch.tensor(pairs, dtype=torch.long)

        # Pre-build every tensor once so __getitem__ is pure indexing
        # (per-sample pandas access dominates training time otherwise).
        scans = torch.tensor(data[scan_columns].values, dtype=torch.float32,
                             device=device)
        goal = torch.tensor(data[goal_columns].values, dtype=torch.float32,
                            device=device)
        target = torch.tensor(data[target_columns].values, dtype=torch.float32,
                              device=device)
        i0, i1 = self.pairs, self.pairs + 1
        self.inputs = torch.cat([scans[i0], scans[i1], goal[i1]], dim=1)
        self.targets = target[i1]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_goal_data(
    file_path,
    scan_columns,
    goal_columns,
    target_columns,
    batch_size=64,
    device="cpu",
):
    data = pd.read_csv(file_path)
    dataset = GoalSenseDataset(data, scan_columns, goal_columns, target_columns, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
