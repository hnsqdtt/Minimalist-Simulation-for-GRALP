import math
import random
import pybullet as p
import pybullet_data
from config_loader import Config


class DynamicObstacle:
    """Box that oscillates along one axis, pausing when the robot is ahead.

    Nominal motion is ``pos = start + sin(eff_t * speed + phase) * range``;
    ``eff_t`` advances by ``dt * slowdown`` per call, so when the robot enters
    the obstacle's forward cone the sinusoid pauses (and resumes once clear).
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

    def update(self, time_t, robot_pos):
        dt = 0.0 if self._prev_t is None else max(0.0, time_t - self._prev_t)
        self._prev_t = time_t

        phase = self._eff_t * self.speed + self.phase
        cur_pos = list(self.start_pos)
        cur_pos[self.axis] += math.sin(phase) * self.range
        # d offset / d eff_t = cos(phase) * speed * range; sign is the heading.
        direction = 1.0 if math.cos(phase) >= 0.0 else -1.0

        self._eff_t += dt * self._slowdown(cur_pos, direction, robot_pos)

        new_phase = self._eff_t * self.speed + self.phase
        new_pos = list(self.start_pos)
        new_pos[self.axis] += math.sin(new_phase) * self.range
        p.resetBasePositionAndOrientation(self.id, new_pos, [0, 0, 0, 1])

    def _slowdown(self, cur_pos, direction, robot_pos):
        # Project the obstacle->robot vector onto the moving axis.
        dx = robot_pos[0] - cur_pos[0]
        dy = robot_pos[1] - cur_pos[1]
        if self.axis == 0:
            forward, lateral = dx * direction, dy
        else:
            forward, lateral = dy * direction, dx
        # Robot is behind us or off to the side: keep the nominal speed.
        if forward <= 0.0 or abs(lateral) > Config.DYNAMIC_OBSTACLE_AVOID_LATERAL:
            return 1.0
        near = Config.DYNAMIC_OBSTACLE_AVOID_NEAR
        far = Config.DYNAMIC_OBSTACLE_AVOID_FAR
        if forward <= near:
            return 0.0
        if forward >= far:
            return 1.0
        return (forward - near) / (far - near)


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
        for _ in range(Config.STATIC_OBSTACLE_COUNT):
            x = random.uniform(-half_map, half_map)
            y = random.uniform(-half_map, half_map)
            if abs(x) < 1.0 and abs(y) < 1.0:
                continue

            size = [random.uniform(0.2, 0.8) for _ in range(3)]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1])
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, size[2]],
            )

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
