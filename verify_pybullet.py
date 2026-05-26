"""Headless verification of the exported mlp_4 policy in the real PyBullet scene.

PART A -- run the policy in the cluttered scene (walls + static + dynamic
          obstacles), headless, and measure whether it heads toward the goal
          and what training-formula reward it would earn here. Run twice:
            A1 faithful  -- exactly as task.py runs it (a negative dilated ray
                            makes inference throw, robot freezes that step);
            A2 clamped   -- negative rays clipped to 0 so inference never
                            throws, isolating the policy from that sim bug.
PART B -- controlled probe: hold the policy fixed, sweep the goal's body angle,
          and check whether the commanded omega steers toward the goal -- under
          (1) training-distribution rays, (2) real pybullet rays, (3) open space.

Run:  python verify_pybullet.py
"""
import math
import random

import numpy as np
import pybullet as p

from config_loader import Config
from policy_runner import PolicyRunner
from robot import Robot
from scene import Scene
import task as T

VX_MAX, OM_MAX, DT = 0.6, 1.5, 0.1


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0


def sample_training_rays(k: int, R: int = 105, rng=np.random) -> np.ndarray:
    """Replicate SimRandomGPUBatchEnv._resample_fov_and_ref (narrow-passage on)."""
    p_empty = np.clip(rng.standard_normal(k) * 10.0, 0.0, 50.0) / 100.0
    mask_empty = rng.random((k, R)) < p_empty[:, None]
    dist = np.clip(np.abs(rng.standard_normal((k, R))) * 1.5, 0.0, 10.0)
    return np.where(mask_empty, 10.0, dist).astype(np.float32)


def part_a(runner: PolicyRunner, clamp: bool, n_control: int = 900, seed: int = 0) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    p.connect(p.DIRECT)
    scene = Scene()
    robot = Robot()
    clearance = Config.ROBOT_RADIUS * 2.5
    goal = T._sample_goal(robot.id, clearance)

    sub = int(round(DT / Config.TIMESTEP))
    prev_vx = prev_om = pp_vx = pp_om = 0.0
    t = 0.0
    herrs, rews, vxs = [], [], []
    ray_pool = []
    infer_fail = neg_steps = coll_steps = goals_reached = 0

    for c in range(n_control):
        pos, yaw = T._world_pose(robot.id)
        raw = np.asarray(robot.get_lidar_data(), dtype=np.float32)
        adjusted, los_pts = T._build_los_points(pos, yaw, raw, runner.patch_meters)
        local = T._select_local_target(los_pts, pos, yaw, (goal[0], goal[1]))
        sin_ref, cos_ref, td = T._direction_to_target(pos, yaw, local)
        td = min(max(td, 0.0), runner.patch_meters)

        dist0 = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        herr = wrap(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - yaw)
        if bool((adjusted < 0.0).any()):
            neg_steps += 1
        rays_in = np.clip(adjusted, 0.0, runner.patch_meters) if clamp else adjusted

        try:
            act = runner.infer(rays_m=rays_in, sin_ref=sin_ref, cos_ref=cos_ref,
                               prev_vx=prev_vx, prev_omega=prev_om,
                               prev_prev_vx=pp_vx, prev_prev_omega=pp_om,
                               task_dist=td, deterministic=True)
            vx, om = float(act[0]), float(act[1])
        except Exception:
            infer_fail += 1
            vx, om = 0.0, 0.0
        ray_pool.append(np.clip(adjusted, 0.0, runner.patch_meters).astype(np.float32))

        collided = False
        for _ in range(sub):
            robot.apply_control(vx, om)
            rpos, _ = T._world_pose(robot.id)
            scene.update(t, rpos)
            p.stepSimulation()
            t += Config.TIMESTEP
            if not collided:
                for cp in p.getContactPoints(bodyA=robot.id):
                    if cp[2] != robot.id and not T._is_plane_body(cp[2]):
                        collided = True
                        break

        new_pos, _ = T._world_pose(robot.id)
        dist1 = math.hypot(goal[0] - new_pos[0], goal[1] - new_pos[1])
        delta_d = dist1 - dist0
        base = -delta_d / (VX_MAX * DT)
        prog = base if (delta_d < 0.0 and vx > 0.0) else -abs(base)
        v_ratio = min(abs(vx) / VX_MAX, 1.0)
        jx = (vx - 2 * prev_vx + pp_vx) / VX_MAX
        jo = (om - 2 * prev_om + pp_om) / OM_MAX
        rew = (0.1 * prog - 2.0 * v_ratio * (1.0 if collided else 0.0)
               - 0.01 * min(jx * jx / 16, 1.0) - 0.01 * min(jo * jo / 16, 1.0))

        herrs.append(abs(herr)); rews.append(rew); vxs.append(vx)
        coll_steps += int(collided)
        pp_vx, pp_om = prev_vx, prev_om
        prev_vx, prev_om = vx, om
        if dist1 <= max(0.5, Config.ROBOT_RADIUS * 1.5):
            goal = T._sample_goal(robot.id, clearance)
            goals_reached += 1

    p.disconnect()
    he = np.array(herrs)
    return dict(reward=float(np.mean(rews)), collision=coll_steps / n_control,
                herr_deg=math.degrees(he.mean()), cos_herr=float(np.cos(he).mean()),
                within90=float((he < math.pi / 2).mean()), mean_vx=float(np.mean(vxs)),
                goals=goals_reached, frozen=infer_fail, neg_steps=neg_steps,
                ray_pool=np.array(ray_pool), n=n_control)


def part_b(runner: PolicyRunner, pyb_rays: np.ndarray):
    print()
    print("=" * 80)
    print(" PART B -- is the policy a goal-tracker, and does it survive OOD rays?")
    print("=" * 80)
    print("  sweep goal body-angle phi; a goal-tracker must turn toward it")
    print("  (sign of omega == sign of phi). probe under 3 ray distributions:\n")
    rng = np.random.default_rng(0)
    K = 512
    phis = np.linspace(-math.pi, math.pi, 49)
    train_mean = sample_training_rays(2000, rng=rng).mean() / 10.0
    pools = {
        f"training-dist rays (norm mean {train_mean:.2f})  [IN-DIST]": "train",
        f"real pybullet rays (norm mean {pyb_rays.mean() / 10:.2f})  [DEPLOY]": "pyb",
        "open space, all rays = max            [reference]": "clear",
    }
    sample_phis = [-135, -90, -45, 0, 45, 90, 135]
    for label, kind in pools.items():
        omega = np.zeros_like(phis)
        vx = np.zeros_like(phis)
        for i, phi in enumerate(phis):
            if kind == "train":
                rays = sample_training_rays(K, rng=rng)
            elif kind == "pyb":
                rays = pyb_rays[rng.integers(0, len(pyb_rays), K)]
            else:
                rays = np.full((K, 105), 10.0, np.float32)
            act = runner.infer(rays_m=rays, sin_ref=math.sin(phi), cos_ref=math.cos(phi),
                               prev_vx=0.0, prev_omega=0.0, prev_prev_vx=0.0,
                               prev_prev_omega=0.0, task_dist=5.0, deterministic=True)
            omega[i] = act[:, 1].mean()
            vx[i] = act[:, 0].mean()
        corr = pearson(omega, phis)
        track = np.mean((np.sign(omega) == np.sign(phis))[phis != 0])
        print(f"  {label}")
        print(f"    corr(omega, phi)        : {corr:+.3f}   (a clean goal-tracker -> ~+1)")
        print(f"    turns toward goal       : {track * 100:.0f}% of swept angles")
        print(f"    mean vx over sweep      : {vx.mean():+.3f} m/s")
        omg = {d: omega[int(np.argmin(np.abs(phis - math.radians(d))))] for d in sample_phis}
        print(f"    omega at phi=[-135..135]: " + " ".join(f"{omg[d]:+.2f}" for d in sample_phis))
        print()


def main():
    runner = PolicyRunner(backend=Config.INFERENCE_BACKEND, device=Config.INFERENCE_DEVICE)
    print(f"[runner] backend={runner.backend}  rays={runner.rays}  obs_dim={runner.obs_dim}\n")
    print("=" * 80)
    print(" PART A -- mlp_4 rollout in the real cluttered PyBullet scene (headless)")
    print("=" * 80)
    a1 = part_a(runner, clamp=False)
    a2 = part_a(runner, clamp=True)
    f = "  {:32s}{:>16}{:>16}"
    print(f.format("", "A1 faithful", "A2 bug-clamped"))
    print(f.format("(matches task.py)", "", ""))
    print(f.format("mean reward / step", f"{a1['reward']:+.5f}", f"{a2['reward']:+.5f}"))
    print(f.format("  [training env: +0.069]", "", ""))
    print(f.format("collision rate (steps)", f"{a1['collision']:.4f}", f"{a2['collision']:.4f}"))
    print(f.format("  [training env: 0.0001]", "", ""))
    print(f.format("frozen steps (infer threw)", f"{a1['frozen']}/{a1['n']}", f"{a2['frozen']}/{a2['n']}"))
    print(f.format("steps w/ negative dilated ray", f"{a1['neg_steps']}/{a1['n']}", f"{a2['neg_steps']}/{a2['n']}"))
    print(f.format("goals reached", str(a1['goals']), str(a2['goals'])))
    print(f.format("mean |heading error| (deg)", f"{a1['herr_deg']:.1f}", f"{a2['herr_deg']:.1f}"))
    print(f.format("mean cos(heading error)", f"{a1['cos_herr']:+.3f}", f"{a2['cos_herr']:+.3f}"))
    print(f.format("fraction within 90 deg", f"{a1['within90']:.3f}", f"{a2['within90']:.3f}"))
    print(f.format("mean vx (m/s)", f"{a1['mean_vx']:+.3f}", f"{a2['mean_vx']:+.3f}"))
    part_b(runner, a1["ray_pool"])


if __name__ == "__main__":
    main()
