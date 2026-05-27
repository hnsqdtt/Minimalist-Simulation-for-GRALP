import json
import os
import random
from types import SimpleNamespace
from typing import Optional

import numpy as np


def load_model_meta(path: Optional[str] = None) -> dict:
    """Load the policy export descriptor written by tools/export_onnx.py."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = path or os.path.join(base_dir, "model", "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_simulation_config(
    sim_config_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> SimpleNamespace:
    """Merge simulation_config.json with the policy contract from model/meta.json.

    Lidar range / ray count and control limits come from meta.json, so the
    simulated observation always matches what the policy was trained on.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = sim_config_path or os.path.join(base_dir, "simulation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = load_model_meta(meta_path)
    obs = meta.get("obs") or {}
    limits = meta.get("limits") or {}
    patch_meters = obs.get("patch_meters")
    rays = obs.get("rays")
    if patch_meters is None or rays is None:
        raise ValueError("model/meta.json must provide obs.patch_meters and obs.rays")

    # Lidar must reproduce the policy's observation contract exactly.
    data.setdefault("LIDAR_FOV", 360)
    data["LIDAR_RANGE"] = float(patch_meters)
    data["LIDAR_NUM_RAYS"] = int(rays)
    data["CTRL_VX_MAX"] = float(limits.get("vx_max", 1.5))
    data["CTRL_OMEGA_MAX"] = float(limits.get("omega_max", 2.0))
    data.setdefault("OBSTACLE_INFLATION_EXTRA", 0.1)
    data.setdefault("INFERENCE_BACKEND", "auto")
    data.setdefault("INFERENCE_DEVICE", "cpu")

    # Translucent orange polygon visualizing the actual line-of-sight visible
    # region (rays clipped at obstacles). Pure visual: built from the same
    # los_points the policy consumes -- zero extra raycasts. No collision
    # shape, mass=0, so it neither blocks rays nor registers contact.
    data.setdefault("SHOW_LOS_POLYGON", True)
    data.setdefault("LOS_POLYGON_COLOR", [1.0, 0.5, 0.0, 0.25])

    speed_range = data.get("DYNAMIC_OBSTACLE_SPEED_RANGE", [0.5, 1.5])
    if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE must be a [min, max] list/tuple")
    speed_min, speed_max = float(speed_range[0]), float(speed_range[1])
    if speed_min > speed_max:
        raise ValueError("DYNAMIC_OBSTACLE_SPEED_RANGE min must be <= max")
    data["DYNAMIC_OBSTACLE_SPEED_RANGE"] = [speed_min, speed_max]

    # Dynamic-obstacle look-ahead (see scene.DynamicObstacle): the obstacle's
    # swept lane is a rectangle of its own perpendicular extent (plus robot
    # radius and LATERAL_CLEARANCE buffer) extending LOOKAHEAD metres past its
    # leading edge; speed ramps linearly from full at LOOKAHEAD down to zero
    # at STOP_CLEARANCE — the surface-to-surface gap at which the obstacle
    # fully stops. LATERAL_CLEARANCE widens the detection lane on both sides
    # so the obstacle still notices a robot grazing past its corner.
    data.setdefault("DYNAMIC_OBSTACLE_LOOKAHEAD", 2.0)
    data.setdefault("DYNAMIC_OBSTACLE_STOP_CLEARANCE", 0.05)
    data.setdefault("DYNAMIC_OBSTACLE_LATERAL_CLEARANCE", 0.05)
    if data["DYNAMIC_OBSTACLE_LOOKAHEAD"] <= 0.0:
        raise ValueError("DYNAMIC_OBSTACLE_LOOKAHEAD must be > 0")
    if data["DYNAMIC_OBSTACLE_STOP_CLEARANCE"] < 0.0:
        raise ValueError("DYNAMIC_OBSTACLE_STOP_CLEARANCE must be >= 0")
    if data["DYNAMIC_OBSTACLE_STOP_CLEARANCE"] >= data["DYNAMIC_OBSTACLE_LOOKAHEAD"]:
        raise ValueError("DYNAMIC_OBSTACLE_STOP_CLEARANCE must be < DYNAMIC_OBSTACLE_LOOKAHEAD")
    if data["DYNAMIC_OBSTACLE_LATERAL_CLEARANCE"] < 0.0:
        raise ValueError("DYNAMIC_OBSTACLE_LATERAL_CLEARANCE must be >= 0")

    # Every TURN_INTERVAL seconds of sim time each obstacle re-randomises its
    # motion axis and initial direction (re-centring on its current position
    # so there is no teleport). 0 or negative disables the periodic turn.
    data.setdefault("DYNAMIC_OBSTACLE_TURN_INTERVAL", 5.0)

    # Wall-clock playback rate (see task.py); does not affect physics or the
    # policy's control dt. 1.0 = real time, <1 = slow motion, >1 = fast.
    data.setdefault("TIME_SCALE", 1.0)
    if data["TIME_SCALE"] <= 0.0:
        raise ValueError("TIME_SCALE must be > 0")

    # On-screen trajectory length cap (one segment per physics step,
    # ~TIMESTEP seconds each). 0 disables the trail entirely.
    data.setdefault("TRAIL_MAX_SEGMENTS", 2000)
    data["TRAIL_MAX_SEGMENTS"] = int(data["TRAIL_MAX_SEGMENTS"])
    if data["TRAIL_MAX_SEGMENTS"] < 0:
        raise ValueError("TRAIL_MAX_SEGMENTS must be >= 0")

    # Seed both stdlib ``random`` and ``numpy.random`` at config load so every
    # downstream draw (scene layout, goal sampling, stochastic inference) is
    # reproducible. ``null`` keeps non-deterministic behaviour.
    seed = data.get("SEED")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
    data["SEED"] = seed

    return SimpleNamespace(**data)


Config = load_simulation_config()
