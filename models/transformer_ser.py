
import torch, torch.nn as nn
from .pooling import AttentionPooling

class ImprovedTransformerSER(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward, num_layers,
                 num_classes, max_len=300, dropout=0.3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(input_dim, dim_feedforward, 3, padding=1),
            nn.BatchNorm1d(dim_feedforward), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim_feedforward))
        layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                           dim_feedforward=dim_feedforward*4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.pool = AttentionPooling(dim_feedforward)
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward), nn.LayerNorm(dim_feedforward),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward//2), nn.LayerNorm(dim_feedforward//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward//2, num_classes)
        )
    def forward(self, x):
        x = self.embed(x)           # (B, D, T)
        x = x.transpose(1,2) + self.pos[:, :x.size(2), :]
        mask = (x.abs().sum(2) == 0)
        x = self.enc(x, src_key_padding_mask=mask)
        x, attn = self.pool(x)
        return self.classifier(x), attn
