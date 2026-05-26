import math
import random
import pybullet as p
import pybullet_data
from config_loader import Config


class DynamicObstacle:
    """Box that oscillates along one axis, pausing when the robot is ahead.

    Nominal motion is ``pos = start + sin(eff_t * speed + phase) * range``;
    ``eff_t`` advances by ``dt * slowdown`` per call, so when the robot enters
    the obstacle's swept lane the sinusoid pauses (and resumes once clear).
    Every ``TURN_INTERVAL`` seconds of sim time, the axis and initial sign are
    re-randomised and ``start`` is re-centred on the current position.
    """

    def __init__(self, pos):
        self.start_pos = list(pos)
        self.size = [0.3, 0.3, 0.5]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in self.size], rgbaColor=[0, 0, 1, 1])
        self.id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )

        self.axis = random.choice([0, 1])
        speed_min, speed_max = Config.DYNAMIC_OBSTACLE_SPEED_RANGE
        self.speed = random.uniform(speed_min, speed_max)
        self.range = 2.0
        self.phase = random.uniform(0, 2 * math.pi)
        self._eff_t = 0.0
        self._prev_t = None
        # Stagger the first turn so the crowd doesn't pivot in lockstep.
        interval = Config.DYNAMIC_OBSTACLE_TURN_INTERVAL
        self._next_turn_t = random.uniform(0.0, interval) if interval > 0.0 else float("inf")

    def update(self, time_t, robot_pos):
        dt = 0.0 if self._prev_t is None else max(0.0, time_t - self._prev_t)
        self._prev_t = time_t

        phase = self._eff_t * self.speed + self.phase
        cur_pos = list(self.start_pos)
        cur_pos[self.axis] += math.sin(phase) * self.range
        # d offset / d eff_t = cos(phase) * speed * range; sign is the heading.
        direction = 1.0 if math.cos(phase) >= 0.0 else -1.0

        # Periodic direction reset on sim time — ignores any slowdown pause.
        if time_t >= self._next_turn_t:
            self._turn(cur_pos)
            self._next_turn_t += Config.DYNAMIC_OBSTACLE_TURN_INTERVAL
            phase = self.phase  # eff_t was reset to 0
            cur_pos = list(self.start_pos)
            direction = 1.0 if math.cos(phase) >= 0.0 else -1.0

        self._eff_t += dt * self._slowdown(cur_pos, direction, robot_pos)

        new_phase = self._eff_t * self.speed + self.phase
        new_pos = list(self.start_pos)
        new_pos[self.axis] += math.sin(new_phase) * self.range
        p.resetBasePositionAndOrientation(self.id, new_pos, [0, 0, 0, 1])

    def _turn(self, cur_pos):
        # Re-centre on the current position and pick a fresh axis + sign so
        # the sinusoid restarts smoothly (sin(0 or pi) == 0 -> no teleport).
        self.start_pos = list(cur_pos)
        self.axis = random.choice([0, 1])
        self.phase = random.choice([0.0, math.pi])
        self._eff_t = 0.0

    def _slowdown(self, cur_pos, direction, robot_pos):
        # Project the obstacle->robot vector onto the moving axis; the swept
        # lane is a rectangle of the obstacle's own width (plus robot radius)
        # extending LOOKAHEAD beyond its leading edge.
        dx = robot_pos[0] - cur_pos[0]
        dy = robot_pos[1] - cur_pos[1]
        if self.axis == 0:
            forward, lateral = dx * direction, dy
        else:
            forward, lateral = dy * direction, dx
        half_along = self.size[self.axis] / 2.0
        half_perp = self.size[1 - self.axis] / 2.0
        r = Config.ROBOT_RADIUS
        # Robot is behind us or outside the swept lane: keep nominal speed.
        if forward <= 0.0 or abs(lateral) > half_perp + r:
            return 1.0
        # Surface-to-surface gap between the obstacle's leading edge and
        # the robot body, along the motion axis. We stop while there is
        # still STOP_CLEARANCE of gap left, so the bodies never quite touch.
        gap = forward - half_along - r
        stop = Config.DYNAMIC_OBSTACLE_STOP_CLEARANCE
        lookahead = Config.DYNAMIC_OBSTACLE_LOOKAHEAD
        if gap <= stop:
            return 0.0
        if gap >= lookahead:
            return 1.0
        return (gap - stop) / (lookahead - stop)


class Scene:
    def __init__(self):
        self.dynamic_obstacles = []
        self._setup_world()
        self._create_walls()
        self._spawn_static_obstacles()
        self._spawn_dynamic_obstacles()

    def _setup_world(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, Config.GRAVITY)
        p.loadURDF("plane.urdf")

    def _create_walls(self):
        half_map = Config.MAP_SIZE / 2
        h_ext = Config.WALL_HEIGHT / 2
        t_ext = Config.WALL_THICKNESS / 2

        walls_info = [
            ([0, half_map + t_ext], [half_map + 2 * t_ext, t_ext]),
            ([0, -half_map - t_ext], [half_map + 2 * t_ext, t_ext]),
            ([-half_map - t_ext, 0], [t_ext, half_map]),
            ([half_map + t_ext, 0], [t_ext, half_map]),
        ]

        for pos_xy, half_size_xy in walls_info:
            pos = [pos_xy[0], pos_xy[1], h_ext]
            size = [half_size_xy[0], half_size_xy[1], h_ext]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.3, 0.3, 0.3, 1])

            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    def _spawn_static_obstacles(self):
        half_map = Config.MAP_SIZE / 2 - 1.0
        # Every pair of static boxes must keep at least this surface-to-surface
        # gap, so the planner-side inflation (ROBOT_RADIUS + EXTRA) still leaves
        # a non-degenerate channel between them.
        min_gap = Config.ROBOT_RADIUS + Config.OBSTACLE_INFLATION_EXTRA
        attempts_per_obstacle = 100
        placed = []  # list of (x, y, half_x, half_y) for placed boxes
        for _ in range(Config.STATIC_OBSTACLE_COUNT):
            for _ in range(attempts_per_obstacle):
                x = random.uniform(-half_map, half_map)
                y = random.uniform(-half_map, half_map)
                if abs(x) < 1.0 and abs(y) < 1.0:
                    continue
                size = [random.uniform(0.2, 0.8) for _ in range(3)]
                hx, hy = size[0], size[1]
                # L2 surface-to-surface distance between two AABBs.
                too_close = False
                for px, py, phx, phy in placed:
                    dx = max(0.0, abs(x - px) - hx - phx)
                    dy = max(0.0, abs(y - py) - hy - phy)
                    if math.hypot(dx, dy) < min_gap:
                        too_close = True
                        break
                if too_close:
                    continue

                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1])
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, size[2]],
                )
                placed.append((x, y, hx, hy))
                break
            # If we never break out, this slot is silently dropped.
        if len(placed) < Config.STATIC_OBSTACLE_COUNT:
            print(f"[scene] placed {len(placed)}/{Config.STATIC_OBSTACLE_COUNT} "
                  f"static obstacles (clearance >= {min_gap:.3f} m)")

    def _spawn_dynamic_obstacles(self):
        half_map = Config.MAP_SIZE / 2 - 1.5
        for _ in range(Config.DYNAMIC_OBSTACLE_COUNT):
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            if abs(x) < 2.0 and abs(y) < 2.0:
                continue
            self.dynamic_obstacles.append(DynamicObstacle([x, y, 0.25]))

    def update(self, time_t, robot_pos):
        for obs in self.dynamic_obstacles:
            obs.update(time_t, robot_pos)
