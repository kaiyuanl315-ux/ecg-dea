"""Exact architecture classes used by the retained-ensemble IG analysis."""
import torch
import torch.nn as nn
from tsai.models.InceptionTime import InceptionTime

class TabMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 64, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalInception(nn.Module):
    """InceptionTime 作为 ECG 编码器 + 表格 MLP，拼接后接 1-logit 头。"""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_tab_in: int,
        *,
        d_ecg_out: int = 256,
        d_tab_out: int = 64,
        nf: int = 32,
        dropout_head: float = 0.1,
    ):
        super().__init__()
        self.ecg_encoder = InceptionTime(c_in=c_in, c_out=d_ecg_out, seq_len=seq_len, nf=nf, residual=True)
        self.tab_encoder = TabMLP(d_in=d_tab_in, d_hidden=d_tab_out, d_out=d_tab_out, p=0.1)
        self.head = nn.Sequential(nn.Dropout(dropout_head), nn.Linear(d_ecg_out + d_tab_out, 1))

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z_ecg = self.ecg_encoder(x_ecg)
        z_tab = self.tab_encoder(x_tab.float())
        z = torch.cat([z_ecg, z_tab], dim=-1)
        return self.head(z)
