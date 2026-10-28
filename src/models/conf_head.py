"""src/models/conf_head.py -- small per-pixel uncertainty head on top of a FROZEN METEOR-Net.

Predicts a Laplace scale b (Kelvin) for each channel/pixel from the input frames, the model's output and
the gap length. The reconstruction network is never changed (its outputs are detached), so Extension 1
cannot degrade Objective-1 accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH = 17      # 2 frames x 4 ch + 4 normalized prediction + 4 correction over linear + 1 gap channel


class ConfHead(nn.Module):
    def __init__(self, in_ch: int = IN_CH, hid: int = 32, out_ch: int = 4, min_scale: float = 0.05):
        super().__init__()
        self.min_scale = min_scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hid, hid, 3, padding=2, dilation=2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hid, hid, 3, padding=4, dilation=4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hid, out_ch, 1),
        )

    def forward(self, x):
        return self.min_scale + F.softplus(self.net(x))


def build_features(st, b, yhat_k):
    """st: ChannelStandardizer, b: to_device batch, yhat_k: [B,4,H,W] Kelvin (detached)."""
    inp = torch.nan_to_num(b["input_norm"].float())                       # [B,4,4,H,W]
    fr = torch.cat([inp[:, 2], inp[:, 3]], dim=1)                          # frames t and t+G
    lin = torch.nan_to_num(0.5 * (b["input_kelvin"][:, 2] + b["input_kelvin"][:, 3]), nan=250.0)
    mean, std = st.mean.reshape(1, -1, 1, 1), st.std.reshape(1, -1, 1, 1)
    yn = (yhat_k - mean) / std
    cor = (yhat_k - lin) / std
    H, W = yhat_k.shape[-2:]
    gap = (b["gap_min"].float() / 30.0).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    return torch.cat([fr, yn, cor, gap], dim=1)


def laplace_nll(y, yhat, scale, mask):
    """Mean Laplace negative log-likelihood (up to a constant) over valid pixels. mask: [B,H,W] bool."""
    m = mask.unsqueeze(1).expand_as(y)
    y = torch.nan_to_num(y, nan=0.0)
    nll = (y - yhat).abs() / scale + torch.log(scale)
    n = m.sum().clamp(min=1)
    return (nll * m).sum() / n
