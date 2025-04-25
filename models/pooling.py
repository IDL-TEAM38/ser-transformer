
import torch, torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x*w).sum(1), w
