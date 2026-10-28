"""src/models/rife_base.py -- RIFE (Practical-RIFE) as an extra interpolation baseline, UNVERIFIED against the real repo.

RIFE was trained on ordinary RGB video, so each brightness-temperature channel is mapped to [0,1] with a FIXED range
(default 170-330 K), repeated to 3 channels, interpolated at t=0.5, averaged back and de-normalised. It uses no
satellite training: that is the point of the baseline ("a strong off-the-shelf video interpolator").

Setup on Colab (see STEP_I_INSTRUCTIONS.md): clone Practical-RIFE, put a pretrained `train_log/` folder
(flownet.pkl + RIFE_HDv3.py + IFNet_HDv3.py ...) inside it. The Practical-RIFE API changed between versions, so this
adapter tries both `inference(a, b, timestep=0.5)` and `inference(a, b)`.
"""
import os
import sys

import torch


class RifeMidpoint:
    def __init__(self, repo_dir: str, device, lo: float = 170.0, hi: float = 330.0):
        sys.path.insert(0, repo_dir)
        from train_log.RIFE_HDv3 import Model          # noqa: E402  (provided by the Practical-RIFE repo)
        self.device, self.lo, self.hi = device, lo, hi
        m = Model()
        folder = os.path.join(repo_dir, "train_log")
        try:
            m.load_model(folder, -1)
        except TypeError:
            m.load_model(os.path.join(folder, "flownet.pkl"), -1, str(device))
        m.eval()
        try:
            m.device()
        except Exception:
            pass
        self.m = m

    @torch.no_grad()
    def __call__(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """xa, xb: [B,C,H,W] Kelvin (H, W multiples of 64) -> [B,C,H,W] Kelvin midpoint."""
        B, C, H, W = xa.shape
        prep = lambda x: ((torch.nan_to_num(x, nan=250.0) - self.lo) / (self.hi - self.lo)).clamp(0, 1).reshape(B * C, 1, H, W).repeat(1, 3, 1, 1)
        a, b = prep(xa).to(self.device), prep(xb).to(self.device)
        outs = []
        for i in range(0, a.shape[0], 32):
            try:
                o = self.m.inference(a[i:i + 32], b[i:i + 32], timestep=0.5)
            except TypeError:
                o = self.m.inference(a[i:i + 32], b[i:i + 32])
            outs.append(o)
        o = torch.cat(outs).mean(dim=1, keepdim=True).reshape(B, C, H, W)
        return o * (self.hi - self.lo) + self.lo
