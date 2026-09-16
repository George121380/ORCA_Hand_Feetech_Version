# 仿真、训练与推理

## 任务定义

下面介绍的是原有的**无物体手势跟踪**示例。新增的接触操作任务见 [掌内转笔](pen.md)：目标是改变笔的朝向并保持不掉落，使用独立的环境、观测、奖励和 checkpoint。

使用官方 `OrcaHandRight(version="v1")`，保留原始几何、碰撞、质量和执行器参数，增加 `GestureTrackingEnv`：

- 目标为张开、握拳、捏合，以及三者的随机凸组合；初始姿态单独随机化。
- 控制周期 50 ms，单回合 100 步，即 5 秒。
- 观测 64 维：16 个手指关节位置、速度、目标位置和上一控制位置；位置按模型范围固定归一化。
- 动作 16 维：手势包络内的绝对关节目标；各手势中恒定的关节保持固定，手腕为模型中立值。
- 奖励为姿态平方误差的指数函数，另有小的动作变化惩罚。
- 成功：平均绝对关节误差不超过 5°，持续 0.5 秒。

任务使用状态观测，无需示教数据。它验证训练与推理流程；没有辨识实际腱绳的摩擦、迟滞和顺应性，示例策略仅部署在仿真。

## 示例模型

```bash
orca-hand infer --checkpoint artifacts/pretrained/gesture-v1/policy.zip \
  --episodes 3 --video --output runs/inference
```

输出 `evaluation.mp4`、`tracking.png`、`metrics.json` 和详细轨迹。checkpoint 与相邻 `config.json` 必须一起移动；载入时检查模型、关节顺序及上游版本。

## 训练与续训

```bash
orca-hand train --steps 120000 --seed 0 --num-envs 4 --output runs/ppo-new
orca-hand train --resume runs/ppo-new/policy.zip --steps 120000 \
  --num-envs 4 --output runs/ppo-continued
tensorboard --logdir runs/ppo-new/tensorboard
```

PPO 使用 128×128 策略／价值网络。默认 CPU、多进程仿真。`--steps` 是跨环境累计步数，实际数向上取整到 rollout 批次，如 120,000 对应 120,832 步。

每 25,000 步附近保存 checkpoint，每 10,000 步附近评估。`best_model.zip` 按评估奖励选择，`policy.zip` 是训练结束的模型。

采用固定归一化，无独立运行统计文件；公式和关节顺序在 `config.json`，范围在包内固定模型元数据中。

## 独立评估

```bash
orca-hand evaluate --checkpoint runs/ppo-new/policy.zip --episodes 100 --output runs/eval-ppo
orca-hand evaluate --baseline --episodes 100 --output runs/eval-baseline
orca-hand evaluate --random --episodes 100 --output runs/eval-random
```

默认测试种子 20000–20099，各方法使用相同初始条件和目标。评估不更新网络。`--video` 保存前三个回合，指标包含全部回合。

连续变化目标可单独训练或评估：

```bash
orca-hand evaluate --checkpoint runs/ppo-new/policy.zip --moving-goal \
  --episodes 100 --output runs/eval-moving
```

此时结合全程误差曲线判断跟踪质量；成功率仍表示曾连续达标 0.5 秒。

复现完整三种子实验：

```bash
.venv/bin/python scripts/benchmark.py --steps 120000 --num-envs 4
```

脚本训练三组模型，分别评估，并加入位置控制与随机策略比较；任一 PPO 成功率低于 90% 时返回非零状态，运行结果保存在本地。

后续可在任务层增加物体保持或掌内旋转，补充物体状态、接触条件、奖励和随机化，再建立仿真与真机动力学一致性。
