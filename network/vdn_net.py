import torch
import torch.nn as nn

class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, agent_qs):
        q_total = torch.sum(agent_qs, dim=-1, keepdim=True)
        return q_total 