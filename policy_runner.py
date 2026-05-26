"""Unified PPO policy runner for the GRALP minimalist simulation.

Loads a policy from a ``model/`` export folder produced by the training repo's
``tools/export_onnx.py``. That folder is self-describing through ``meta.json``;
this runner picks the ``.onnx`` (onnxruntime) or ``.pt`` (torch) backend and
exposes a single ``infer()`` that turns metric lidar rays + pose into a
``(vx, omega)`` command.

Observation contract (see README.md): ``obs = [rays_norm(R), pose(7)]``, with
``obs_dim = R + 7``. ``rays_norm`` is the metric ranges divided by
``patch_meters``; the 7-dim pose tail is
``[sin_ref, cos_ref, prev_vx/vx_max, prev_omega/omega_max,
   dvx/(2*vx_max), domega/(2*omega_max), task_dist/patch_meters]``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]

POSE_DIM = 7
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent / "model_config.json"


# --------------------------------------------------------------------------
# meta.json / model_config.json
# --------------------------------------------------------------------------
def load_meta(model_dir: Union[str, Path]) -> dict:
    """Load and lightly validate ``model_dir/meta.json``."""
    meta_path = Path(model_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {model_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("encoder", "obs", "limits", "dt", "action_dim", "log_std_min", "log_std_max"):
        if key not in meta:
            raise ValueError(f"meta.json is missing required key {key!r}")
    return meta


def _validate_against_schema(meta: dict, model_configs: dict) -> None:
    """Check meta.json's encoder kind and params against the model_config.json schema."""
    spec = meta["encoder"]
    name = spec.get("name")
    params = spec.get("params") or {}
    same_kind = [e for e in model_configs.values()
                 if isinstance(e, dict) and e.get("name") == name]
    if not same_kind:
        kinds = sorted({e.get("name") for e in model_configs.values() if isinstance(e, dict)})
        raise ValueError(
            f"meta.json encoder kind {name!r} is not declared in model_config.json (known: {kinds})")
    required: set = set()
    for entry in same_kind:
        required |= set((entry.get("params") or {}).keys())
    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"meta.json encoder.params is missing {sorted(missing)} required by model_config.json")


# --------------------------------------------------------------------------
# observation assembly (shared by both backends)
# --------------------------------------------------------------------------
def _ensure_2d_rays(rays_m: ArrayLike) -> np.ndarray:
    arr = np.asarray(rays_m, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"rays_m must be 1-D [R] or 2-D [B, R], got {arr.ndim}-D")


def _normalize_rays(rays_m: np.ndarray, patch_meters: float) -> np.ndarray:
    """Validate metric rays lie in [0, patch_meters], return them scaled to [0, 1]."""
    if not np.isfinite(rays_m).all():
        raise ValueError("rays_m contains non-finite values")
    if (rays_m < 0.0).any():
        raise ValueError("rays_m must be >= 0")
    if (rays_m > patch_meters).any():
        raise ValueError(f"rays_m must be <= patch_meters ({patch_meters})")
    return (rays_m / patch_meters).astype(np.float32)


def _build_pose(batch: int, *, sin_ref, cos_ref, prev_vx, prev_omega, prev_prev_vx,
                prev_prev_omega, task_dist, vx_max, omega_max, patch_meters) -> np.ndarray:
    """Assemble the 7-dim pose tail; every field is a scalar or a length-``batch`` array."""
    def as_b(x, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((batch,), float(arr), dtype=np.float32)
        elif arr.shape == (batch, 1):
            arr = arr.reshape(batch)
        elif arr.shape != (batch,):
            raise ValueError(f"{name}: expected a scalar or shape [{batch}], got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        return arr

    sin_v = as_b(sin_ref, "sin_ref")
    cos_v = as_b(cos_ref, "cos_ref")
    if (np.abs(sin_v) > 1.0 + 1e-4).any() or (np.abs(cos_v) > 1.0 + 1e-4).any():
        raise ValueError("sin_ref / cos_ref must lie in [-1, 1]")
    if not np.allclose(sin_v * sin_v + cos_v * cos_v, 1.0, atol=5e-2):
        raise ValueError("sin_ref^2 + cos_ref^2 must be ~1 (tolerance 0.05)")

    prev_vx_v = as_b(prev_vx, "prev_vx")
    prev_om_v = as_b(prev_omega, "prev_omega")
    pp_vx_v = as_b(prev_prev_vx, "prev_prev_vx")
    pp_om_v = as_b(prev_prev_omega, "prev_prev_omega")
    td_v = as_b(task_dist, "task_dist")
    if (td_v < 0.0).any() or (td_v > patch_meters).any():
        raise ValueError(f"task_dist must lie in [0, patch_meters={patch_meters}]")

    pose = np.stack([
        sin_v,
        cos_v,
        prev_vx_v / max(vx_max, 1e-9),
        prev_om_v / max(omega_max, 1e-9),
        (prev_vx_v - pp_vx_v) / max(2.0 * vx_max, 1e-9),
        (prev_om_v - pp_om_v) / max(2.0 * omega_max, 1e-9),
        td_v / max(patch_meters, 1e-9),
    ], axis=-1)
    return pose.astype(np.float32)


# --------------------------------------------------------------------------
# backends — each maps obs [B, obs_dim] -> (mu, log_std), both [B, action_dim]
# --------------------------------------------------------------------------
def _select_onnx_providers(ort, device: Optional[str]) -> list:
    available = list(ort.get_available_providers())
    pref = (device or "cpu").strip().lower()
    if pref.startswith(("tensorrt", "trt", "tensor")):
        order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif pref.startswith(("cuda", "gpu")):
        order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        order = ["CPUExecutionProvider"]
    return [p for p in order if p in available] or available


class _OnnxBackend:
    """onnxruntime backend — the graph is self-contained; model_config is not needed."""

    kind = "onnx"

    def __init__(self, onnx_path: Path, device: Optional[str]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("onnxruntime is required for the .onnx backend "
                              "(`pip install onnxruntime` or `onnxruntime-gpu`)") from exc
        self.session = ort.InferenceSession(
            str(onnx_path), providers=_select_onnx_providers(ort, device))

    def run(self, obs: np.ndarray, limits: np.ndarray):
        mu, log_std = self.session.run(["mu", "log_std"], {"obs": obs, "limits": limits})
        return np.asarray(mu, dtype=np.float32), np.asarray(log_std, dtype=np.float32)


class _TorchBackend:
    """torch backend — rebuilds the network from meta.json, validated against model_config.json."""

    kind = "pt"

    def __init__(self, pt_path: Path, meta: dict, model_config_path: Path,
                 device: Optional[str]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("torch is required for the .pt backend") from exc
        from gralp_net import build_policy

        if not model_config_path.is_file():
            raise FileNotFoundError(
                f"model_config.json not found at {model_config_path}; required by the .pt backend")
        model_configs = json.loads(model_config_path.read_text(encoding="utf-8"))
        _validate_against_schema(meta, model_configs)

        policy = build_policy(meta["encoder"], obs_dim=int(meta["obs"]["obs_dim"]),
                              action_dim=int(meta["action_dim"]),
                              log_std_min=float(meta["log_std_min"]),
                              log_std_max=float(meta["log_std_max"]))
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        state = payload["policy"] if isinstance(payload, dict) and "policy" in payload else payload
        try:
            policy.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"policy.pt does not match the network described by meta.json "
                f"(encoder {meta['encoder'].get('name')!r}): {exc}") from exc

        want_cuda = bool(device) and "cuda" in device.lower()
        self.device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
        self.policy = policy.to(self.device).eval()
        self._torch = torch

    def run(self, obs: np.ndarray, limits: np.ndarray):
        with self._torch.no_grad():
            t_obs = self._torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
            mu, log_std = self.policy(t_obs)
        return mu.cpu().numpy().astype(np.float32), log_std.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
class PolicyRunner:
    """Loads a ``model/`` export folder and runs deterministic / sampled inference.

    Parameters
    ----------
    model_dir : export folder; defaults to ``./model`` next to this file.
    backend   : ``"auto"`` (onnx if present, else pt), ``"onnx"``, or ``"pt"``.
    device    : ``"cpu"`` (default), ``"cuda"``, or ``"tensorrt"`` (onnx backend).
    model_config_path : ``model_config.json`` used by the ``.pt`` backend;
                        defaults to ``model_config.json`` next to this file.
    """

    def __init__(self, model_dir: Union[str, Path, None] = None, *,
                 backend: str = "auto", device: Optional[str] = None,
                 model_config_path: Union[str, Path, None] = None) -> None:
        model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        meta = load_meta(model_dir)
        self.meta = meta

        obs = meta["obs"]
        self.patch_meters = float(obs["patch_meters"])
        self.ray_max_gap = float(obs["ray_max_gap"])
        self.rays = int(obs["rays"])
        self.obs_dim = int(obs["obs_dim"])
        self.pose_dim = int(obs.get("pose_dim", POSE_DIM))
        self.vx_max = float(meta["limits"]["vx_max"])
        self.omega_max = float(meta["limits"]["omega_max"])
        self.dt = float(meta["dt"])
        self.action_dim = int(meta["action_dim"])

        onnx_path = model_dir / "policy.onnx"
        pt_path = model_dir / "policy.pt"
        chosen = self._choose_backend(backend, onnx_path, pt_path)
        if chosen == "onnx":
            self._backend = _OnnxBackend(onnx_path, device)
        else:
            mc_path = Path(model_config_path) if model_config_path is not None \
                else DEFAULT_MODEL_CONFIG
            self._backend = _TorchBackend(pt_path, meta, mc_path, device)
        self.backend = self._backend.kind

    @staticmethod
    def _choose_backend(backend: str, onnx_path: Path, pt_path: Path) -> str:
        backend = (backend or "auto").lower()
        if backend == "onnx":
            if not onnx_path.is_file():
                raise FileNotFoundError(f"backend='onnx' but {onnx_path} is missing")
            return "onnx"
        if backend == "pt":
            if not pt_path.is_file():
                raise FileNotFoundError(f"backend='pt' but {pt_path} is missing")
            return "pt"
        if backend == "auto":
            if onnx_path.is_file():
                return "onnx"
            if pt_path.is_file():
                return "pt"
            raise FileNotFoundError(f"no policy.onnx or policy.pt in {onnx_path.parent}")
        raise ValueError(f"backend must be 'auto', 'onnx', or 'pt'; got {backend!r}")

    def infer(self, *, rays_m: ArrayLike, sin_ref: ArrayLike, cos_ref: ArrayLike,
              prev_vx: ArrayLike, prev_omega: ArrayLike,
              prev_prev_vx: ArrayLike, prev_prev_omega: ArrayLike,
              task_dist: ArrayLike, deterministic: bool = True) -> np.ndarray:
        """Run one (or a batch of) inference step(s); returns actions in SI units.

        ``rays_m`` are metric lidar ranges, 1-D ``[R]`` or 2-D ``[B, R]``, with
        ray 0 aligned with the robot heading. Returns ``[vx, omega]`` for a
        single frame, or ``[B, 2]`` for a batch.
        """
        single = np.asarray(rays_m).ndim == 1
        rays = _ensure_2d_rays(rays_m)
        batch, n_rays = int(rays.shape[0]), int(rays.shape[1])
        if n_rays != self.rays:
            raise ValueError(f"rays_m length {n_rays} != expected ray count {self.rays}")

        rays_n = _normalize_rays(rays, self.patch_meters)
        pose = _build_pose(batch, sin_ref=sin_ref, cos_ref=cos_ref,
                           prev_vx=prev_vx, prev_omega=prev_omega,
                           prev_prev_vx=prev_prev_vx, prev_prev_omega=prev_prev_omega,
                           task_dist=task_dist, vx_max=self.vx_max,
                           omega_max=self.omega_max, patch_meters=self.patch_meters)
        obs = np.concatenate([rays_n, pose], axis=-1).astype(np.float32)
        limits = np.tile(np.array([self.vx_max, self.omega_max], dtype=np.float32), (batch, 1))

        mu, log_std = self._backend.run(obs, limits)
        if deterministic:
            pre_tanh = mu
        else:
            eps = np.random.standard_normal(mu.shape).astype(np.float32)
            pre_tanh = mu + np.exp(log_std) * eps
        action = (np.tanh(pre_tanh) * limits).astype(np.float32)

        return action[0] if single else action
