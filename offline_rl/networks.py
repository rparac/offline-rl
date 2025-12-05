from torch import nn
import torch.nn.functional as F
import torch

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MultiDiscrete([4, 4]) is input
        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 1)

    def forward(self, obs):
        # x = torch.cat([obs[:, 0], obs[:, 1]], dim=-1).float()
        x = self.lin1(obs)
        x = self.lin2(x)
        return x

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MultiDiscrete([4, 4]) is input
        self.num_actions = 4
        self.obs_embed = nn.Linear(2, 16)
        self.to_act = nn.Linear(16, self.num_actions)

    def forward(self, obs):
        x = self.obs_embed(obs)
        x = self.to_act(x)
        return x

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MultiDiscrete([4, 4]) is input
        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 4)

    def forward(self, obs):
        squeezed = False
        if obs.dim() == 1:
            # non-batched input
            obs = obs.unsqueeze(0)
            squeezed = True


        # x = torch.cat([obs[:, 0], obs[:, 1]], dim=-1).float()
        x = self.lin1(obs)
        x = self.lin2(x)

        if squeezed:
            x = x.squeeze(0)

        return x