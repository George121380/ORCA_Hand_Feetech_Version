# 软件架构与接口

## 组织方式

- `src/orca_feetech/`：项目自己的模型元数据、控制、STS 通信、标定、视频输入、任务和 CLI。
- `.upstream/`：按提交下载的官方源码和模型；通过 `uv` 作为依赖安装，不修改其文件。
- `configs/`：不含实测值的配置模板；`local/` 保存每台设备的标定及生成的重定向配置。
- `runs/`：完整运行数据；`artifacts/` 保存小型示例模型、可审查的验证摘要。

## 关节坐标

内部采用 **v1 右手 MJCF 的 qpos 坐标，单位 rad**。17 个关节按拇指、食指、中指、无名指、小指、手腕排序。MuJoCo 把手腕执行器排在首位，因此执行器和状态索引全部按名称建立映射。

```text
URDF 关节角 = MJCF qpos − 对应 joint.ref
OrcaJointPositions 的角度值 = rad2deg(MJCF qpos)
舵机编码值 = 每台实体手的三点分段线性标定(MJCF qpos)
```

上述角度接口是本项目明确采用的模型坐标，不能直接混入原始 Dynamixel 标定的角度／零位。包内范围与中立值从锁定的官方 MJCF 提取；三点标定把真实关节对齐到这些模型坐标。

自动化测试逐一比较 URDF 与 MJCF 的 18 个共同节点的位置和旋转，覆盖基础手势与随机姿态。

## 控制与状态

`JointTarget` 包含 `positions`、`joint_names`、`model_id` 和单调时钟时间戳。必须提供完整的 17 个有限值；顺序、版本或值不合法会报错。

`HandState` 包含关节位置、反馈时间、`estimated` 标志和总线遥测。真机的关节位置由舵机反馈换算，`estimated=True`。

三个后端共用 `read / write / render / close`。CLI 的完整目标经过 `CommandGuard` 限位和速度限制；原始 RGB 输入的时间戳跨过求解阶段保留，慢求解不会刷新旧数据的年龄。

预设动作采用最小跃度五次多项式。异常停止优先停止目标推进，可能打断正常加速度曲线。真机后端另有独立看门狗，主循环阻塞时仍检查输入超时。

## 使用官方 BaseHand API

`OrcaHandController` 继承官方 `BaseHand`，保留具名角度命令、状态和姿态管理接口，通过项目后端控制 STS 或仿真。

```python
from orca_feetech.api import OrcaHandController

with OrcaHandController("mock") as hand:
    hand.move_to_pose("pinch")
    print(hand.get_joint_position().as_dict())
```

流式调用 `set_joint_positions` 时提供度数值；此方法只推进一帧，完整平滑动作使用 `move_to_pose`。部分关节更新保持其他关节目标不变。主流程的网络／视频边界使用更严格的完整 `JointTarget`。

硬件 API 使用 `OrcaHandController("hardware", hardware_config="local/hand.yaml")`。首版没有调用官方 `OrcaHand.init_joints()`，因为它会自动设置电流并触发标定，与 STS3215 的位置控制路径不符。

## 记录与兼容性

轨迹为 JSONL，旁边的 `metadata.json` 记录格式版本、模型、关节名、单位、采样率和运行状态。只有完整运行的轨迹允许回放。回放重新建立当前时钟时间，从实读姿态平滑衔接；不沿用录制机器的单调时钟数值。

PPO 环境是独立的 Gymnasium 任务，共用模型和命名映射。策略仅连接仿真，不提供一键实机策略部署；未来需要先辨识实际执行器与腱绳动力学。
