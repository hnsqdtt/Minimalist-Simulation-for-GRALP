# Minimalist Simulation for GRALP

极简 PyBullet 仿真环境，用于可视化验证 [GRALP](https://github.com/hnsqdtt/GRALP) 训练出的 PPO 局部规划策略。策略以一个自包含的 `model/` 文件夹形式提供（由 GRALP 主仓库的 `tools/export_onnx.py` 导出），仿真侧的 `PolicyRunner` 可直接加载其中的 `.onnx` 或 `.pt`。

## 依赖

- Python 3.8+
- `pybullet`、`numpy` —— 仿真本体
- `onnxruntime` —— `.onnx` 后端（默认）；需要 GPU/TensorRT 时改装 `onnxruntime-gpu`
- `torch` —— 仅 `.pt` 后端需要

```bash
pip install pybullet numpy onnxruntime
# 若要用 .pt 后端，再按设备安装 torch
```

## 快速开始

1. 确认仓库根目录有 `model/` 文件夹（见下节；由 GRALP 主仓库导出）。
2. 运行：`python task.py`
3. 相机：右键旋转视角，左键拖拽平移，滚轮缩放；`Ctrl+C` 退出。

## 策略模型：`model/` 文件夹

`model/` 是 GRALP 主仓库 `tools/export_onnx.py` 导出的自包含产物：

| 文件 | 内容 |
|---|---|
| `meta.json` | 自描述文件：编码器种类、结构参数，以及 obs/动作契约（`patch_meters`、`ray_max_gap`、`limits`、`dt` 等） |
| `policy.onnx` | ONNX 计算图；输入 `obs`/`limits`，输出 `action`/`mu`/`log_std` |
| `policy.pt` | PyTorch 检查点（原始权重） |

**替换模型**：在 GRALP 主仓库重新跑 `python tools/export_onnx.py --tag <run>`（或 `--ckpt <xxx.pt>`），把生成的 `model/` 文件夹整体替换进来即可。仿真侧的雷达量程、射线数、控制限幅都会从新的 `meta.json` 自动派生，无需改代码。

**权重与网络如何配合**：
- `.onnx` 后端 —— 计算图已内含结构与权重，`onnxruntime` 直接执行，不需要 `model_config.json`。
- `.pt` 后端 —— `policy.pt` 是裸权重，需要网络结构定义：`PolicyRunner` 读 `meta.json` 拿到编码器种类与参数，用 `gralp_net.py` 重建网络，再把权重以 `strict` 模式加载进去；`model_config.json` 用于校验 `meta.json` 的种类与字段是否合法。

**后端选择**（`simulation_config.json` 的 `INFERENCE_BACKEND`）：

| 值 | 行为 |
|---|---|
| `auto`（默认） | 有 `policy.onnx` 就用 onnx，否则用 `policy.pt` |
| `onnx` | onnxruntime；只依赖 `onnxruntime`，最轻量 |
| `pt` | torch；依据 `meta.json` + `model_config.json` 重建网络再加载，需要 `torch` |

`INFERENCE_DEVICE`：`cpu`（默认）/ `cuda` / `tensorrt`，是 onnx 后端的执行设备。

## 配置文件

| 文件 | 作用 | 是否常改 |
|---|---|---|
| `simulation_config.json` | 纯仿真场景：物理、地图、障碍、相机、机器人外观，以及 `INFERENCE_BACKEND` / `INFERENCE_DEVICE` | 常改 |
| `model_config.json` | 模型结构 schema（4 种编码器的字段定义），`.pt` 后端用它校验 `meta.json` | 一般不动 |

> obs 契约（`patch_meters`、`ray_max_gap`、`vx_max`、`omega_max`、`dt`）**不在**这两个文件里 —— 它们随权重走，存放在 `model/meta.json`。`config_loader.py` 读 `meta.json` 派生出 `LIDAR_RANGE`、`LIDAR_NUM_RAYS`、`CTRL_VX_MAX`、`CTRL_OMEGA_MAX`，从而保证仿真采集的观测与策略训练时严格一致。

## 观测与动作：仿真如何与策略配合

每个控制周期（`meta.dt` 秒）仿真侧依次做这些事：

1. **采集雷达** —— `robot.get_lidar_data()` 发出 `LIDAR_NUM_RAYS` 条射线，射线 0 对齐车头，其余按等角度逆时针铺满 360°，返回米制命中距离。
2. **LOS 膨胀** —— `task._build_los_points()` 把命中点按车体半径向外膨胀、裁到 `patch_meters` 内，得到送给策略的 `adjusted_ranges`；另按 `ROBOT_RADIUS + OBSTACLE_INFLATION_EXTRA` 膨胀一份用于选目标。
3. **折射局部目标** —— `task._select_local_target()` 把全局目标投影到可见线段上，取最接近全局目标的可行点；完全遮挡时沿当前航向取前向可行距离。
4. **构造方向特征** —— `task._direction_to_target()` 在车体坐标系算出局部目标方向单位向量 `(sin_ref, cos_ref)` 与距离 `task_dist`。
5. **策略推理** —— `PolicyRunner.infer()` 接收上述量，内部归一化、拼装观测向量、前向，得到 `[vx, omega]`。
6. **执行** —— `robot.apply_control()` 按 `CTRL_VX_MAX` / `CTRL_OMEGA_MAX` 限幅后写入 PyBullet。

### 观测规范

观测向量 `obs` 维度为 `R + 7`，由 `PolicyRunner` 内部拼装，调用方无需自己拼：

```
obs = [ rays_norm(R) , pose(7) ]
```

**射线段 `rays_norm`（R 维）**

| 项 | 规范 |
|---|---|
| 长度 R | 见 `meta.json` 的 `obs.rays`，等于 `ceil(2π·patch_meters / ray_max_gap)` |
| 单位 | 米；传入 `infer()` 的 `rays_m` 为米制原始距离 |
| 取值范围 | `[0, patch_meters]`，越界或非有限值会报错 |
| 角度布局 | 射线 0 对齐车体朝向，其余等角度铺满 360° |
| 归一化 | `PolicyRunner` 内部除以 `patch_meters`，得到 `[0, 1]` |

**姿态尾部 `pose`（7 维）** —— 由 `PolicyRunner` 从传入分量构造：

| 序号 | 分量 | 含义 |
|---|---|---|
| 0 | `sin_ref` | 局部目标方向（车体系）的正弦 |
| 1 | `cos_ref` | 局部目标方向（车体系）的余弦；要求 `sin² + cos² ≈ 1` |
| 2 | `prev_vx / vx_max` | 上一帧线速度，归一化 |
| 3 | `prev_omega / omega_max` | 上一帧角速度，归一化 |
| 4 | `(prev_vx − prev_prev_vx) / (2·vx_max)` | 线速度一阶差分 |
| 5 | `(prev_omega − prev_prev_omega) / (2·omega_max)` | 角速度一阶差分 |
| 6 | `task_dist / patch_meters` | 局部目标距离，归一化 |

### 动作规范

`infer()` 返回 `[vx, omega]`，SI 单位：

- `vx ∈ [−vx_max, vx_max]`，`omega ∈ [−omega_max, omega_max]`
- 确定性动作 = `tanh(mu) · limits`；传 `deterministic=False` 则按训练分布采样

## PolicyRunner 接口

```python
import numpy as np
from policy_runner import PolicyRunner

# 加载 ./model 文件夹;backend 取 "auto" | "onnx" | "pt"
runner = PolicyRunner(backend="auto", device="cpu")

# 单帧推理
action = runner.infer(
    rays_m=np.full(runner.rays, 10.0),       # 米制射线,长度必须 == runner.rays
    sin_ref=0.0, cos_ref=1.0,                # 局部目标方向(车体系单位向量)
    prev_vx=0.0,  prev_omega=0.0,            # 上一帧动作
    prev_prev_vx=0.0, prev_prev_omega=0.0,   # 上上帧动作
    task_dist=5.0,                           # 局部目标距离(米),∈ [0, patch_meters]
    deterministic=True,
)
print(action)        # -> [vx, omega]

# 批量推理:rays_m 传 [B, R],其余分量传标量或长度 B 的数组,返回 [B, 2]
```

构造后可直接读取的契约属性：`runner.rays`、`runner.obs_dim`、`runner.patch_meters`、`runner.vx_max`、`runner.omega_max`、`runner.dt`、`runner.backend`。

**严格校验**（不满足即抛异常）：

- `rays_m` 长度必须等于 `runner.rays`
- `rays_m` 每个值有限且 ∈ `[0, patch_meters]`
- `sin_ref`、`cos_ref` ∈ `[-1, 1]`，且 `sin² + cos² ≈ 1`（容差 0.05）
- `task_dist` ∈ `[0, patch_meters]`
- `.pt` 后端：`meta.json` 的编码器种类与必要字段须与 `model_config.json` 对齐，权重须与重建的网络结构匹配，否则报错

## 局部任务点选择机制

- **全局目标** —— `task._sample_goal` 在地图内随机采样，与障碍/墙体保持 2.5×车体半径的安全距离；机器人接近后自动重采样下一个目标。
- **任务折射** —— 全局目标经上文步骤 2–3 投影到当前可见线段，得到一个始终可达的局部目标点送入策略。
- **可视化** —— 目标十字、局部连线、碰撞变色、运动轨迹均由 `task.py` 用 `addUserDebugLine/Text` 绘制；`simulation_config.json` 的 `DEBUG_MODE` 为 `true` 时额外显示雷达射线。

## 常用调整

- 增减 `STATIC_OBSTACLE_COUNT` / `DYNAMIC_OBSTACLE_COUNT`，或修改 `DYNAMIC_OBSTACLE_SPEED_RANGE` 来压测策略。
- 调高 `WALL_HEIGHT` 或 `MAP_SIZE` 观察大场景下的表现。
- 修改 `CAM_DIST` / `CAM_YAW` / `CAM_PITCH` 设定初始视角，或 `MOUSE_SENSITIVITY_*` 改变交互手感。
- 更换策略：替换整个 `model/` 文件夹（见上文「策略模型」）。
