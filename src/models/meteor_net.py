"""
src/models/meteor_net.py
Flow-free, sequence-based temporal super-resolution architecture based on METEOR-Net (Lin et al., IEEE TGRS 2026).
Adapted for single-target midpoint reconstruction [t-60, t-30, t, t+30] -> t+15 min on GOES-19 / INSAT-3DS infrared data.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalChannelSqueeze(nn.Module):
    """
    Global Channel Squeeze (GCS) block.
    Extracts cross-channel descriptors via global average pooling and generates
    channel gating vector s in [0, 1]^C to project fused multichannel features.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        s = self.channel_gate(self.avg_pool(x))  # [B, C_in, 1, 1]
        x_gated = x * s                          # [B, C_in, H, W]
        out = self.proj(x_gated)                 # [B, C_out, H, W]
        return out


class LCEBlock(nn.Module):
    """
    Local Convolutional Encoding (LCE) block.
    Consists of Conv2d + GroupNorm(32) + LeakyReLU(0.2).
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 64, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class MultichannelCollaborativeEncoding(nn.Module):
    """
    MCE Module:
    - Sequence Branch (3 layers): Encodes 4 sequence frames, downsampling to latent space via Stride 2 at L1.
    - Skip Branch (1 layer): Encodes boundary frames at t=0 and t=+30 at native spatial resolution.
    - Skip Blend: 1x1 Conv blending boundary skip features into the midpoint skip highway.
    """
    def __init__(self, in_channels: int = 4, feat_channels: int = 64):
        super().__init__()
        # Shared GCS across all time steps
        self.gcs = GlobalChannelSqueeze(in_channels, feat_channels)
        
        # Sequence Branch: 3 layers (Layer 1 has stride 2 for latent space compression)
        self.seq_lce1 = LCEBlock(feat_channels, feat_channels, stride=2)
        self.seq_lce2 = LCEBlock(feat_channels, feat_channels, stride=1)
        self.seq_lce3 = LCEBlock(feat_channels, feat_channels, stride=1)
        
        # Skip Branch: 1 layer at native spatial resolution
        self.skip_lce = LCEBlock(feat_channels, feat_channels, stride=1)
        
        # Skip Feature Blend (Learned 1x1 Conv)
        self.skip_blend = nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1, bias=True)

    def forward(
        self,
        seq_frames: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_frames: [B, T=4, C_in=4, H=128, W=128]
            
        Returns:
            seq_latents: List of 4 latent tensors [B, 64, 64, 64]
            anchor_x0: Latent feature at t=0 (index 2), shape [B, 64, 64, 64]
            skip_midpoint: Blended skip feature at t+15 min, shape [B, 64, 128, 128]
        """
        B, T, C, H, W = seq_frames.shape
        
        seq_latents = []
        for t in range(T):
            frame = seq_frames[:, t, :, :, :]
            f_gcs = self.gcs(frame)              # [B, 64, 128, 128]
            x_seq = self.seq_lce1(f_gcs)         # [B, 64, 64, 64]
            x_seq = self.seq_lce2(x_seq)         # [B, 64, 64, 64]
            x_seq = self.seq_lce3(x_seq)         # [B, 64, 64, 64]
            seq_latents.append(x_seq)
            
        anchor_x0 = seq_latents[2]  # Frame 3 is anchor t=0
        
        # Boundary frames for midpoint t+15 min: Frame 2 (t=0) and Frame 3 (t=+30 min)
        frame_0 = seq_frames[:, 2, :, :, :]
        frame_1 = seq_frames[:, 3, :, :, :]
        
        skip_0 = self.skip_lce(self.gcs(frame_0))  # [B, 64, 128, 128]
        skip_1 = self.skip_lce(self.gcs(frame_1))  # [B, 64, 128, 128]
        
        skip_cat = torch.cat([skip_0, skip_1], dim=1)  # [B, 128, 128, 128]
        skip_midpoint = self.skip_blend(skip_cat)      # [B, 64, 128, 128]
        
        return seq_latents, anchor_x0, skip_midpoint


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell with Layer Normalization on internal gates.
    Combines input and recurrent states into a single gate convolution for efficiency.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Single combined gate convolution for [i, f, c_cand, o]
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        # Layer Normalization per gate: 4 separate GroupNorm(1, hidden_channels) blocks
        self.ln_i = nn.GroupNorm(1, hidden_channels)
        self.ln_f = nn.GroupNorm(1, hidden_channels)
        self.ln_c = nn.GroupNorm(1, hidden_channels)
        self.ln_o = nn.GroupNorm(1, hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        
        # Split into 4 gates: [B, hidden_channels, H, W] each
        cc = self.hidden_channels
        i_gate, f_gate, c_cand, o_gate = torch.split(gates, cc, dim=1)
        
        i = torch.sigmoid(self.ln_i(i_gate))
        f = torch.sigmoid(self.ln_f(f_gate))
        c_tilde = torch.tanh(self.ln_c(c_cand))
        o = torch.sigmoid(self.ln_o(o_gate))
        
        c_next = f * c_prev + i * c_tilde
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LowResFlowEstimation(nn.Module):
    """
    4-layer stacked ConvLSTM network for temporal-flow estimation across 4 time steps.
    Produces 4 multiscale horizontal temporal flow tensors {F^1_low, ..., F^4_low}.
    """
    def __init__(self, in_channels: int = 64, hidden_channels: int = 256):
        super().__init__()
        self.layer1 = ConvLSTMCell(in_channels=in_channels, hidden_channels=hidden_channels)
        self.layer2 = ConvLSTMCell(in_channels=hidden_channels, hidden_channels=hidden_channels)
        self.layer3 = ConvLSTMCell(in_channels=hidden_channels, hidden_channels=hidden_channels)
        self.layer4 = ConvLSTMCell(in_channels=hidden_channels, hidden_channels=hidden_channels)
        self.hidden_channels = hidden_channels

    def forward(self, seq_latents: List[torch.Tensor]) -> List[torch.Tensor]:
        B, _, H, W = seq_latents[0].shape
        device = seq_latents[0].device
        
        # Initialize recurrent states
        h1 = c1 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        h2 = c2 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        h3 = c3 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        h4 = c4 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        
        # Recurrent processing across the 4 sequence steps
        for x_t in seq_latents:
            h1, c1 = self.layer1(x_t, (h1, c1))
            h2, c2 = self.layer2(h1, (h2, c2))
            h3, c3 = self.layer3(h2, (h3, c3))
            h4, c4 = self.layer4(h3, (h4, c4))
            
        # Accumulated horizontal temporal flows after step 4
        flows = [h1, h2, h3, h4]
        return flows


class SingleTargetEvolutionPyramid(nn.Module):
    """
    Single-target vertical evolution pyramid.
    Couples anchor feature X_0 with multiscale horizontal flows {F^1_low, ..., F^4_low}
    to produce exactly one intermediate midpoint latent feature Y_{t_mid} in R^{B x 64 x 64 x 64}.
    """
    def __init__(self, anchor_channels: int = 64, flow_channels: int = 256, out_channels: int = 64):
        super().__init__()
        # Layer 1: [X_0; F^1_low] -> 64 + 256 = 320 -> 256
        self.conv1 = nn.Conv2d(anchor_channels + flow_channels, flow_channels, kernel_size=3, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=flow_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layers 2, 3, 4: [E^{n-1}; F^n_low] -> 256 + 256 = 512 -> 256
        self.conv2 = nn.Conv2d(flow_channels * 2, flow_channels, kernel_size=3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=flow_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(flow_channels * 2, flow_channels, kernel_size=3, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=flow_channels)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(flow_channels * 2, flow_channels, kernel_size=3, padding=1, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=flow_channels)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        
        # Top projection to latent evolved dimension (64 channels)
        self.top_proj = nn.Conv2d(flow_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, anchor_x0: torch.Tensor, flows: List[torch.Tensor]) -> torch.Tensor:
        # Layer 1
        e1 = self.act1(self.gn1(self.conv1(torch.cat([anchor_x0, flows[0]], dim=1))))
        # Layer 2
        e2 = self.act2(self.gn2(self.conv2(torch.cat([e1, flows[1]], dim=1))))
        # Layer 3
        e3 = self.act3(self.gn3(self.conv3(torch.cat([e2, flows[2]], dim=1))))
        # Layer 4
        e4 = self.act4(self.gn4(self.conv4(torch.cat([e3, flows[3]], dim=1))))
        
        y_mid = self.top_proj(e4)  # [B, 64, 64, 64]
        return y_mid


class TemporalFlowEstimationAndEvolution(nn.Module):
    """
    TFEE Module:
    Combines Low-Resolution Flow Estimation (ConvLSTM) and Single-Target Evolution Pyramid.
    """
    def __init__(self, in_channels: int = 64, hidden_channels: int = 256, out_channels: int = 64):
        super().__init__()
        self.flow_estimator = LowResFlowEstimation(in_channels, hidden_channels)
        self.evolution_pyramid = SingleTargetEvolutionPyramid(in_channels, hidden_channels, out_channels)

    def forward(self, seq_latents: List[torch.Tensor], anchor_x0: torch.Tensor) -> torch.Tensor:
        flows = self.flow_estimator(seq_latents)
        y_mid = self.evolution_pyramid(anchor_x0, flows)
        return y_mid


class EvolutionAwareFeatureFusionAndReconstruction(nn.Module):
    """
    EFFR Module:
    1. Reconstruction Submodule: Restores spatial dimensions 64x64 -> 128x128 via ConvTranspose2d.
    2. Summation with Skip Feature: I = X'_{t_mid} + ~Y_{t_mid}.
    3. Decoupled Attention Maps:
       - c_map in R^{B x 64 x 1 x 1}
       - s_map in R^{B x 1 x 128 x 128}
       - e_map = sigma(W_p * concat(c_map + s_map, I)) in [0, 1]^{B x 64 x 128 x 128}
    4. Dynamic Redistribution: ~I = I + e_map * ~Y + (1 - e_map) * X'.
    5. Scale Restoration: Maps 64 feature channels back to 4 calibrated observation channels.
    """
    def __init__(self, feat_channels: int = 64, out_channels: int = 4):
        super().__init__()
        # 1. Reconstruction Submodule
        self.recon_conv = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=True)
        self.recon_gn1 = nn.GroupNorm(num_groups=32, num_channels=feat_channels)
        self.recon_act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.recon_deconv = nn.ConvTranspose2d(
            feat_channels, feat_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.recon_gn2 = nn.GroupNorm(num_groups=32, num_channels=feat_channels)
        self.recon_act2 = nn.LeakyReLU(0.2, inplace=True)

        # 2. Decoupled Attention Blocks
        # Channel Attention Map (c_map)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_mlp = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 4, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels, kernel_size=1, bias=True)
        )
        # Spatial Attention Map (s_map): 7x7 conv on [mean_c; max_c] -> 1 channel
        self.s_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        
        # Evolution-Aware Attention Map (e_map):
        # Input: concat(c_map + s_map, I) -> 64 + 64 = 128 channels. Output: 64 channels.
        self.e_conv = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid()
        )

        # 3. Scale Restoration
        self.h_conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=True)
        self.h_gn = nn.GroupNorm(num_groups=32, num_channels=feat_channels)
        self.h_act = nn.LeakyReLU(0.2, inplace=True)
        self.h_conv1 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, y_mid: torch.Tensor, skip_midpoint: torch.Tensor) -> torch.Tensor:
        # y_mid: [B, 64, 64, 64]
        # skip_midpoint: [B, 64, 128, 128]
        
        # 1. Spatial Dimension Restoration
        y_recon = self.recon_act1(self.recon_gn1(self.recon_conv(y_mid)))
        y_tilde = self.recon_act2(self.recon_gn2(self.recon_deconv(y_recon)))  # [B, 64, 128, 128]

        # 2. Skip Summation
        I_mid = skip_midpoint + y_tilde  # [B, 64, 128, 128]

        # 3. Decoupled Attention Maps
        # Channel weight map
        c_map = self.c_mlp(self.avg_pool(I_mid))  # [B, 64, 1, 1]
        
        # Spatial weight map
        mean_c = torch.mean(I_mid, dim=1, keepdim=True)  # [B, 1, 128, 128]
        max_c, _ = torch.max(I_mid, dim=1, keepdim=True)  # [B, 1, 128, 128]
        s_map = self.s_conv(torch.cat([mean_c, max_c], dim=1))  # [B, 1, 128, 128]
        
        # Broadcast addition: [B, 64, 1, 1] + [B, 1, 128, 128] -> [B, 64, 128, 128]
        attn_sum = c_map + s_map
        
        # Concatenate with I_mid: [B, 64 + 64, 128, 128] = [B, 128, 128, 128]
        e_input = torch.cat([attn_sum, I_mid], dim=1)
        e_map = self.e_conv(e_input)  # [B, 64, 128, 128]

        # 4. Dynamic Redistribution
        I_tilde = I_mid + e_map * y_tilde + (1.0 - e_map) * skip_midpoint

        # 5. Scale Restoration to Calibrated Output Space
        h_feat = self.h_act(self.h_gn(self.h_conv2(I_tilde)))
        h_out = self.h_conv1(h_feat)  # [B, 4, 128, 128]
        return h_out


class METEORNet(nn.Module):
    """
    End-to-End Adapted METEOR-Net Architecture:
    MCE -> TFEE -> EFFR for 4-channel midpoint super-resolution on GOES-19 / INSAT-3DS.
    
    Inputs:
      x: [B, 4, 4, 128, 128] (Batch, 4 Frames at [-60, -30, 0, +30] min, 4 IR Channels, 128x128)
    Outputs:
      out: [B, 4, 128, 128] (Batch, 4 IR Channels at target midpoint t+15 min, 128x128)
    """
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: int = 64,
        tfee_hidden_channels: int = 256,
        out_channels: int = 4
    ):
        super().__init__()
        self.mce = MultichannelCollaborativeEncoding(in_channels=in_channels, feat_channels=feat_channels)
        self.tfee = TemporalFlowEstimationAndEvolution(
            in_channels=feat_channels,
            hidden_channels=tfee_hidden_channels,
            out_channels=feat_channels
        )
        self.effr = EvolutionAwareFeatureFusionAndReconstruction(
            feat_channels=feat_channels,
            out_channels=out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Multichannel Collaborative Encoding
        seq_latents, anchor_x0, skip_midpoint = self.mce(x)
        
        # 2. Temporal-Flow Estimation and Single-Target Evolution
        y_mid = self.tfee(seq_latents, anchor_x0)
        
        # 3. Evolution-Aware Fusion & Spatial Scale Restoration
        h_out = self.effr(y_mid, skip_midpoint)
        return h_out
