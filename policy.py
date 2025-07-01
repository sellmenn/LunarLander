import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        return self.model(obs)
    
def get_parameters(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)

def set_parameters(model, flat_params):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.tensor(flat_params[i:i+n].reshape(p.shape), dtype=torch.float32).to(p.device)
        i += n