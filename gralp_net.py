"""GRALP policy networks — inference-only, self-contained.

A trimmed single-file copy of the training repo's ``models/`` package: the four
encoder kinds (``gralp_attn`` / ``cnn_circular`` / ``cnn_zeropad`` / ``mlp``) and
a ``PPOPolicy`` head. Module and parameter names are kept identical to the
training repo, so a checkpoint exported there loads here with ``strict=True``.

Only the forward path needed for ``.pt`` inference is kept — no training-time
squashing, log-prob, or optimizer logic.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# padding helpers
# --------------------------------------------------------------------------
def circular_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Wrap the sequence end-to-end so the ring of rays stays continuous."""
    if pad <= 0:
        return x
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


def zero_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    return F.pad(x, (pad, pad), mode="constant", value=0.0)


class EncoderBase(nn.Module):
    """Common contract: vec [B, vec_dim] -> feature [B, feature_dim]."""

    def __init__(self, vec_dim: int, *, feature_dim: int = 256) -> None:
        super().__init__()
        self.vec_dim = int(vec_dim)
        self.feature_dim = int(feature_dim)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------
# gralp_attn: circular dilated conv branch + multi-query, multi-head attention
# --------------------------------------------------------------------------
class _SqueezeExcite1D(nn.Module):
    def __init__(self, ch: int, r: int = 4) -> None:
        super().__init__()
        hid = max(8, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class _DepthwiseSeparable1D(nn.Module):
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, groups=ch, bias=False, dilation=self.dil)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = circular_pad1d(x, pad)
        out = self.dw(x)
        out = F.gelu(out)
        out = self.pw(out)
        return self.bn(out)


class _RayBranch(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5) -> None:
        super().__init__()
        self.expand = nn.Conv1d(int(in_ch), hidden, kernel_size=1)
        blocks = []
        for d in [1, 2, 4, 8][:layers]:
            blocks += [
                _DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d),
                nn.GELU(),
                _SqueezeExcite1D(hidden, r=4),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.blocks(self.expand(x))


class GRALPAttnEncoder(EncoderBase):
    """Ray encoder with circular dilated conv + multi-query, multi-head attention."""

    def __init__(self, vec_dim: int, *, feature_dim: int = 256, hidden: int = 64,
                 d_model: int = 128, num_queries: int = 4, num_heads: int = 4,
                 learnable_queries: bool = True, pose_dim: int = 7) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        self.pose_dim = int(pose_dim)
        if self.vec_dim < self.pose_dim:
            raise ValueError(f"vec_dim must be >= pose_dim ({self.pose_dim}), got {vec_dim}")
        self.N = max(0, self.vec_dim - self.pose_dim)
        self.ray_in_ch = 1
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        if self.d_model % max(1, self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.br_obs = _RayBranch(in_ch=self.ray_in_ch, hidden=self.hidden)
        self.to_k = nn.Conv1d(self.hidden, self.d_model, kernel_size=1)
        self.to_v = nn.Conv1d(self.hidden, self.d_model, kernel_size=1)
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, self.d_model), nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if self.learnable_queries:
            init_scale = 1.0 / math.sqrt(max(1, self.d_model))
            self.q_params = nn.Parameter(torch.randn(self.num_queries, self.d_model) * init_scale)
            self.to_q = nn.Identity()
        else:
            if self.num_queries > 1:
                self.to_q = nn.Linear(self.d_model, self.d_model * self.num_queries)
            else:
                self.to_q = nn.Identity()
        self.post = nn.Sequential(
            nn.Linear(self.d_model * 3, self.feature_dim), nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        )

    def _split(self, vec: torch.Tensor):
        d_obs = vec[:, :self.N]
        pose = vec[:, self.N:self.N + self.pose_dim]
        return d_obs, pose

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        d_obs, pose = self._split(vec)
        Fmap = self.br_obs(d_obs)
        K = self.to_k(Fmap).transpose(1, 2)
        V = self.to_v(Fmap).transpose(1, 2)

        q_pose = self.pose_mlp(pose)
        if self.learnable_queries:
            q = self.q_params.unsqueeze(0) + q_pose.unsqueeze(1)
        elif self.num_queries > 1:
            qM = self.to_q(q_pose)
            q = qM.view(qM.size(0), self.num_queries, self.d_model)
        else:
            q = q_pose.view(q_pose.size(0), 1, self.d_model)

        H = max(1, self.num_heads)
        Dh = self.d_model // H
        K_h = K.view(K.size(0), K.size(1), H, Dh)
        V_h = V.view(V.size(0), V.size(1), H, Dh)
        Q_h = q.view(q.size(0), q.size(1), H, Dh)

        attn_logits = torch.einsum("bmhd,bnhd->bmhn", Q_h, K_h) / math.sqrt(Dh)
        attn = torch.softmax(attn_logits, dim=-1)
        z_h = torch.einsum("bmhn,bnhd->bmhd", attn, V_h)
        z = z_h.reshape(z_h.size(0), z_h.size(1), self.d_model)

        z_mean = z.mean(dim=1)
        q_mean = q.mean(dim=1)
        gavg = V.mean(dim=1)
        g = torch.cat([z_mean, gavg, q_mean], dim=-1)
        return self.post(g)


# --------------------------------------------------------------------------
# cnn_circular / cnn_zeropad: dilated 1D-CNN over the ray ring
# --------------------------------------------------------------------------
class DilatedConv1DBlock(nn.Module):
    """Conv1d + BN + GELU with explicit circular or zero padding."""

    def __init__(self, ch: int, *, kernel: int = 5, dilation: int = 1,
                 padding_mode: str = "circular") -> None:
        super().__init__()
        if padding_mode not in ("circular", "zero"):
            raise ValueError(f"padding_mode must be 'circular' or 'zero', got {padding_mode!r}")
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(ch, ch, kernel_size=self.kernel, dilation=self.dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dilation) // 2
        x = circular_pad1d(x, pad) if self.padding_mode == "circular" else zero_pad1d(x, pad)
        return F.gelu(self.bn(self.conv(x)))


class RayPoseCNN1D(EncoderBase):
    """Shared 1D-CNN backbone for ray distances + pose features."""

    PADDING_MODE = "circular"

    def __init__(self, vec_dim: int, *, feature_dim: int = 256, channels: int = 32,
                 kernel: int = 5, dilations: tuple = (1, 2, 4, 8, 16),
                 pose_dim: int = 7, pose_hidden: int = 64) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        self.pose_dim = int(pose_dim)
        self.N = self.vec_dim - self.pose_dim
        if self.N <= 0:
            raise ValueError(f"vec_dim={vec_dim} too small for pose_dim={pose_dim}")
        self.expand = nn.Conv1d(1, int(channels), kernel_size=1)
        self.blocks = nn.Sequential(*[
            DilatedConv1DBlock(int(channels), kernel=int(kernel), dilation=int(d),
                               padding_mode=self.PADDING_MODE)
            for d in dilations
        ])
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, int(pose_hidden)), nn.ReLU(),
            nn.Linear(int(pose_hidden), int(pose_hidden)), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(channels) + int(pose_hidden), self.feature_dim), nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        rays = vec[:, :self.N].unsqueeze(1)
        pose = vec[:, self.N:self.N + self.pose_dim]
        ray_feat = self.blocks(self.expand(rays)).mean(dim=-1)
        pose_feat = self.pose_mlp(pose)
        return self.fuse(torch.cat([ray_feat, pose_feat], dim=-1))


class CNNCircularEncoder(RayPoseCNN1D):
    """1D CNN with circular padding — the ring-topology-aware ablation arm."""

    PADDING_MODE = "circular"


class CNNZeroPadEncoder(RayPoseCNN1D):
    """1D CNN with zero padding — the baseline that ignores ring topology."""

    PADDING_MODE = "zero"


# --------------------------------------------------------------------------
# mlp: plain fully-connected encoder over the flat observation vector
# --------------------------------------------------------------------------
class MLPEncoder(EncoderBase):
    """``depth`` Linear layers (>= 2); the last projects to ``feature_dim``."""

    def __init__(self, vec_dim: int, *, feature_dim: int = 256,
                 hidden: int = 256, depth: int = 3) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        if depth < 2:
            raise ValueError(f"MLPEncoder requires depth >= 2, got {depth}")
        layers = []
        in_dim = self.vec_dim
        for _ in range(int(depth) - 1):
            layers += [nn.Linear(in_dim, int(hidden)), nn.ReLU()]
            in_dim = int(hidden)
        layers += [nn.Linear(in_dim, self.feature_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.net(vec)


# --------------------------------------------------------------------------
# policy head
# --------------------------------------------------------------------------
class PPOPolicy(nn.Module):
    """Gaussian policy head over a pluggable encoder — inference subset.

    Yields ``(mu, log_std)``; action squashing is left to the caller. The value
    head is built (though unused at inference) so a full training checkpoint
    loads with ``strict=True``.
    """

    def __init__(self, encoder: EncoderBase, action_dim: int = 2, *,
                 value_hidden: int = 256, log_std_min: float = -5.0,
                 log_std_max: float = 2.0) -> None:
        super().__init__()
        self.encoder = encoder
        feat = int(encoder.feature_dim)
        self.mu = nn.Linear(feat, int(action_dim))
        self.log_std = nn.Parameter(torch.zeros(int(action_dim)))
        lo, hi = sorted((float(log_std_min), float(log_std_max)))
        self._log_std_min, self._log_std_max = lo, hi
        self.value = nn.Sequential(
            nn.Linear(feat, int(value_hidden)), nn.ReLU(),
            nn.Linear(int(value_hidden), 1),
        )

    def forward(self, obs_vec: torch.Tensor):
        """Observation batch [B, obs_dim] -> (mu, log_std), each [B, action_dim]."""
        feat = self.encoder(obs_vec)
        mu = self.mu(feat)
        log_std = self.log_std.view(1, -1).expand_as(mu).clamp(self._log_std_min, self._log_std_max)
        return mu, log_std


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
ENCODER_REGISTRY: dict[str, type] = {
    "mlp": MLPEncoder,
    "cnn_zeropad": CNNZeroPadEncoder,
    "cnn_circular": CNNCircularEncoder,
    "gralp_attn": GRALPAttnEncoder,
}


def build_encoder(spec: Mapping[str, Any], vec_dim: int) -> EncoderBase:
    """Instantiate an encoder from a ``{name, params}`` mapping."""
    name = spec.get("name")
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; supported: {sorted(ENCODER_REGISTRY)}")
    params = dict(spec.get("params") or {})
    return ENCODER_REGISTRY[name](vec_dim=int(vec_dim), **params)


def build_policy(encoder_spec: Mapping[str, Any], *, obs_dim: int, action_dim: int = 2,
                 log_std_min: float = -5.0, log_std_max: float = 2.0) -> PPOPolicy:
    """Build a ``PPOPolicy`` from an encoder spec, ready for a strict state_dict load."""
    encoder = build_encoder(encoder_spec, obs_dim)
    return PPOPolicy(encoder, action_dim=int(action_dim),
                     log_std_min=float(log_std_min), log_std_max=float(log_std_max))
