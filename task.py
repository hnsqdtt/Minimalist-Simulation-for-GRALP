import math
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p

from camera_ctrl import CameraController
from config_loader import Config
from policy_runner import PolicyRunner
from robot import Robot
from scene import Scene


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _body_name(body_id: int) -> str:
    try:
        info = p.getBodyInfo(body_id)
    except Exception:
        return ""
    if not info or len(info) < 2:
        return ""
    try:
        return info[1].decode("utf-8")
    except Exception:
        return ""


def _point_to_aabb_distance(pt: Vec3, aabb: Tuple[Vec3, Vec3]) -> float:
    (xmin, ymin, zmin), (xmax, ymax, zmax) = aabb
    dx = max(xmin - pt[0], 0.0, pt[0] - xmax)
    dy = max(ymin - pt[1], 0.0, pt[1] - ymax)
    dz = max(zmin - pt[2], 0.0, pt[2] - zmax)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _collect_obstacle_aabbs(robot_id: int) -> List[Tuple[int, Tuple[Vec3, Vec3]]]:
    aabbs: List[Tuple[int, Tuple[Vec3, Vec3]]] = []
    for bid in range(p.getNumBodies()):
        if bid == robot_id:
            continue
        name = _body_name(bid)
        if "plane" in name:
            continue
        try:
            aabb = p.getAABB(bid)
        except Exception:
            continue
        if aabb is None or len(aabb) != 2:
            continue
        aabbs.append((bid, aabb))
    return aabbs


def _is_free(candidate_xy: Vec2, robot_id: int, clearance: float) -> bool:
    aabbs = _collect_obstacle_aabbs(robot_id)
    pt3 = (candidate_xy[0], candidate_xy[1], 0.0)
    for _, aabb in aabbs:
        if _point_to_aabb_distance(pt3, aabb) < clearance:
            return False
    return True


def _sample_goal(robot_id: int, clearance: float) -> Vec3:
    half_map = Config.MAP_SIZE / 2.0 - clearance
    for _ in range(200):
        cand = (
            random.uniform(-half_map, half_map),
            random.uniform(-half_map, half_map),
            0.0,
        )
        if _is_free((cand[0], cand[1]), robot_id, clearance):
            return cand
    raise RuntimeError("Failed to sample a collision-free goal position.")


def _draw_goal_marker(goal: Vec3, existing_ids: Sequence[int]) -> List[int]:
    for gid in existing_ids:
        try:
            p.removeUserDebugItem(gid)
        except Exception:
            pass
    color = [0.2, 1.0, 0.6]
    arm = 0.35
    height = 0.05
    center = [goal[0], goal[1], height]
    arms = [
        ([-arm, 0, 0], [arm, 0, 0]),
        ([0, -arm, 0], [0, arm, 0]),
    ]
    ids: List[int] = []
    for a_start, a_end in arms:
        start = [center[0] + a_start[0], center[1] + a_start[1], center[2] + a_start[2]]
        end = [center[0] + a_end[0], center[1] + a_end[1], center[2] + a_end[2]]
        ids.append(p.addUserDebugLine(start, end, color, lineWidth=3, lifeTime=0))
    ids.append(
        p.addUserDebugLine(
            [goal[0], goal[1], 0.0],
            [goal[0], goal[1], 0.6],
            color,
            lineWidth=3,
            lifeTime=0,
        )
    )
    ids.append(
        p.addUserDebugText(
            "GOAL",
            [goal[0], goal[1], 0.7],
            textColorRGB=color,
            textSize=1.5,
            lifeTime=0,
        )
    )
    return ids


def _world_pose(body_id: int) -> Tuple[Vec3, float]:
    pos, orn = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(orn)
    return (float(pos[0]), float(pos[1]), float(pos[2])), float(yaw)


def _build_los_points(
    pos: Vec3, yaw: float, raw_ranges: np.ndarray, patch_limit: float
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, float]]]:
    fov_rad = math.radians(Config.LIDAR_FOV)
    angle_step = fov_rad / Config.LIDAR_NUM_RAYS
    raw_ranges = np.asarray(raw_ranges, dtype=np.float32)
    dir_cache = []
    for idx in range(Config.LIDAR_NUM_RAYS):
        ang = yaw + idx * angle_step
        dir_cache.append((math.cos(ang), math.sin(ang)))

    def _dilate_ranges(inflate_radius: float) -> np.ndarray:
        dilated = np.minimum(raw_ranges, patch_limit).astype(np.float32)
        if inflate_radius <= 0.0:
            return dilated

        rx, ry, _ = pos
        radius_sq = inflate_radius * inflate_radius

        for i, hit_dist in enumerate(raw_ranges):
            hit_dist = float(hit_dist)
            if not math.isfinite(hit_dist) or hit_dist >= patch_limit - 1e-6:
                continue
            dir_ix, dir_iy = dir_cache[i]
            hx = rx + hit_dist * dir_ix
            hy = ry + hit_dist * dir_iy
            vec_x = hx - rx
            vec_y = hy - ry
            base_len_sq = vec_x * vec_x + vec_y * vec_y

            for j, (dir_jx, dir_jy) in enumerate(dir_cache):
                proj = vec_x * dir_jx + vec_y * dir_jy  # signed distance along ray j
                if proj < 0.0:
                    continue
                cross_sq = base_len_sq - proj * proj  # squared lateral distance from ray j to hit point
                if cross_sq >= radius_sq:
                    continue
                offset = math.sqrt(max(0.0, radius_sq - cross_sq))
                inter = proj - offset  # first intersection with the inflated disk along ray j
                if inter < dilated[j]:
                    dilated[j] = inter
        return dilated

    # PPO dilation uses robot radius; LOS adds inflation.
    rx, ry, _ = pos
    ppo_dilated = _dilate_ranges(max(0.0, Config.ROBOT_RADIUS))
    los_dilated = (
        ppo_dilated
        if math.isclose(
            Config.ROBOT_RADIUS, Config.ROBOT_RADIUS + Config.OBSTACLE_INFLATION_EXTRA, rel_tol=0.0, abs_tol=1e-9
        )
        else _dilate_ranges(max(0.0, Config.ROBOT_RADIUS + Config.OBSTACLE_INFLATION_EXTRA))
    )

    los_points: List[Tuple[float, float, float, float, float]] = []
    adjusted = []
    for idx, dist in enumerate(los_dilated):
        dist = min(float(dist), patch_limit)
        dir_x, dir_y = dir_cache[idx]
        lx = rx + dist * dir_x
        ly = ry + dist * dir_y
        los_points.append((lx, ly, dist, dir_x, dir_y))
    for dist in ppo_dilated:
        adjusted.append(min(float(dist), patch_limit))
    return np.asarray(adjusted, dtype=np.float32), los_points


def _select_local_target(
    los_points: List[Tuple[float, float, float, float, float]], robot_pos: Vec3, robot_yaw: float, goal_xy: Vec2
) -> Tuple[float, float, float]:
    if not los_points:
        return (goal_xy[0], goal_xy[1], 0.0)
    gx, gy = goal_xy
    rx, ry, _ = robot_pos

    heading_x = math.cos(robot_yaw)
    heading_y = math.sin(robot_yaw)

    overlap = None
    for _, _, dist, dir_x, dir_y in los_points:
        if dist < 0.0 and (overlap is None or dist > overlap[0]):
            overlap = (dist, dir_x, dir_y)

    if overlap is not None:
        best_dir = None
        best_dir_dist = None
        best_dot = -float("inf")
        for _, _, dist, dir_x, dir_y in los_points:
            if dist > 0.0:
                dot = dir_x * heading_x + dir_y * heading_y
                if dot > best_dot:
                    best_dot = dot
                    best_dir = (dir_x, dir_y)
                    best_dir_dist = dist
        if best_dir is None:
            best_dir = (heading_x, heading_y)
            best_dir_dist = abs(overlap[0])
        return (rx + best_dir_dist * best_dir[0], ry + best_dir_dist * best_dir[1], best_dir_dist)

    best_point = None
    best_d2 = float("inf")
    for _, _, dist, dir_x, dir_y in los_points:
        gd_x = gx - rx
        gd_y = gy - ry
        t = gd_x * dir_x + gd_y * dir_y  # dot(goal-rel, dir)
        t = max(0.0, min(dist, t))
        px = rx + t * dir_x
        py = ry + t * dir_y
        d2 = (px - gx) ** 2 + (py - gy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_point = (px, py, t)
    return best_point if best_point is not None else (goal_xy[0], goal_xy[1], 0.0)


def _is_plane_body(body_id: int) -> bool:
    try:
        name = p.getBodyInfo(body_id)[1].decode("utf-8")
    except Exception:
        return False
    return "plane" in name.lower()


def _direction_to_target(robot_pos: Vec3, robot_yaw: float, target: Tuple[float, float, float]) -> Tuple[float, float, float]:
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 1.0, 0.0
    # Align to robot frame to match lidar rays.
    ang = math.atan2(dy, dx) - robot_yaw
    cos_ref = math.cos(ang)
    sin_ref = math.sin(ang)
    return sin_ref, cos_ref, dist


def main() -> None:
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    scene = Scene()
    robot = Robot()
    cam_ctrl = CameraController()
    runner = PolicyRunner(backend=Config.INFERENCE_BACKEND, device=Config.INFERENCE_DEVICE)

    clearance = Config.ROBOT_RADIUS * 2.5
    goal = _sample_goal(robot.id, clearance)
    goal_dbg = _draw_goal_marker(goal, [])

    prev_vx = 0.0
    prev_omega = 0.0
    prev_prev_vx = 0.0
    prev_prev_omega = 0.0
    hold_vx = 0.0
    hold_omega = 0.0
    control_interval = float(runner.dt)
    next_control_time = 0.0

    local_dbg: List[int] = []
    trail_dbg: List[int] = []
    trail_max_len = Config.TRAIL_MAX_SEGMENTS
    robot_is_green = False

    t = 0.0
    try:
        while True:
            # Run control at PPO dt.
            while t + 1e-9 >= next_control_time:
                robot_pos, yaw = _world_pose(robot.id)
                raw_lidar = robot.get_lidar_data()
                ranges = np.asarray(raw_lidar, dtype=np.float32)
                adjusted_ranges, los_points = _build_los_points(robot_pos, yaw, ranges, runner.patch_meters)
                robot.update_los_polygon(robot_pos, los_points)

                local_target = _select_local_target(los_points, robot_pos, yaw, (goal[0], goal[1]))
                sin_ref, cos_ref, task_dist = _direction_to_target(robot_pos, yaw, local_target)
                task_dist = min(task_dist, runner.patch_meters)

                try:
                    action = runner.infer(
                        rays_m=adjusted_ranges,
                        sin_ref=sin_ref,
                        cos_ref=cos_ref,
                        prev_vx=prev_vx,
                        prev_omega=prev_omega,
                        prev_prev_vx=prev_prev_vx,
                        prev_prev_omega=prev_prev_omega,
                        task_dist=task_dist,
                        deterministic=True,
                    )
                    prev_prev_vx, prev_prev_omega = prev_vx, prev_omega
                    prev_vx = float(action[0])
                    prev_omega = float(action[1])
                    hold_vx, hold_omega = prev_vx, prev_omega
                except Exception as exc:
                    print(f"[warn] PPO inference failed: {exc}")
                    hold_vx, hold_omega = 0.0, 0.0

                next_control_time += control_interval

                for did in local_dbg:
                    try:
                        p.removeUserDebugItem(did)
                    except Exception:
                        pass
                local_dbg = [
                    p.addUserDebugLine(
                        [robot_pos[0], robot_pos[1], robot_pos[2] + 0.05],
                        [local_target[0], local_target[1], robot_pos[2] + 0.05],
                        [1, 0.6, 0.1],
                        lifeTime=control_interval,
                    )
                ]

                goal_dist = math.hypot(goal[0] - robot_pos[0], goal[1] - robot_pos[1])
                if goal_dist <= max(0.5, Config.ROBOT_RADIUS * 1.5):
                    goal = _sample_goal(robot.id, clearance)
                    goal_dbg = _draw_goal_marker(goal, goal_dbg)

            robot.apply_control(hold_vx, hold_omega)

            # Collision highlight: turn robot green when contacting any non-plane body.
            collided = False
            for cp in p.getContactPoints(bodyA=robot.id):
                other = cp[2]
                if other == robot.id or _is_plane_body(other):
                    continue
                collided = True
                break
            if collided != robot_is_green:
                target_color = [0.1, 0.9, 0.1, 1.0] if collided else Config.ROBOT_COLOR
                p.changeVisualShape(robot.id, -1, rgbaColor=target_color)
                robot_is_green = collided

            robot_pos, _ = _world_pose(robot.id)

            # Draw trajectory segment; darker colour for higher speed.
            # trail_max_len == 0 disables the trail entirely.
            if trail_max_len > 0:
                speed_mag = min(math.hypot(hold_vx, hold_omega), 1.0)
                base_color = np.array([0.2, 0.8, 1.0], dtype=np.float32)
                color = (base_color * (0.3 + 0.7 * speed_mag)).tolist()
                if "last_trail_pos" not in locals():
                    last_trail_pos = robot_pos
                else:
                    trail_dbg.append(
                        p.addUserDebugLine(
                            [last_trail_pos[0], last_trail_pos[1], last_trail_pos[2] + 0.02],
                            [robot_pos[0], robot_pos[1], robot_pos[2] + 0.02],
                            color,
                            lifeTime=0,
                            lineWidth=2,
                        )
                    )
                    last_trail_pos = robot_pos
                    if len(trail_dbg) > trail_max_len:
                        try:
                            p.removeUserDebugItem(trail_dbg.pop(0))
                        except Exception:
                            pass

            scene.update(t, robot_pos)
            cam_ctrl.update()
            p.stepSimulation()
            robot.update_heading_indicator()

            time.sleep(Config.TIMESTEP / Config.TIME_SCALE)
            t += Config.TIMESTEP

    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
