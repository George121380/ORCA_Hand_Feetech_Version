# 掌内转笔：从关节跟踪到物体操作

## 之前的 PPO 在做什么

`GestureTrackingEnv` 给出一组目标关节角，让手指到达目标。观测中直接包含这些角度，奖励只衡量关节跟踪误差。它能验证训练、存档、载入和推理，但没有物体，也没有学习抓取或转笔。

## 这次的任务

`PenReorientationEnv` 使用官方 ORCA v1 掌心向上的模型。把一支约 **14.4 cm 长、14 mm 粗、18 g** 的胶囊形训练笔放在手上，红色一端用于区分朝向。机器人需要将笔的长轴转向左或右 **90°**，同时留在手上。

- **物理**：笔具有自由的六维运动，由重力、摩擦和真实碰撞接触推进；没有焊接约束、悬浮力或每帧直接修改笔的姿态。
- **动作**：16 个手指关节的目标增量，速度限制 0.8 rad/s，手腕保持中立；动作范围覆盖官方关节范围。
- **观测**：69 维，包括手指状态、上一目标、笔的位置／长轴／速度、目标朝向及六组手部接触标志。物体状态来自仿真真值。
- **奖励**：转向目标的进展、方向接近程度、保持接触；惩罚偏移、过大的动作与掉落。
- **成功**：方向误差 <10°、距起始位置 <4 cm、保持手部接触、线速度 <0.1 m/s、角速度 <1 rad/s，连续满足 0.5 秒。
- **掉落**：笔中心低于 8 cm，或离开起始位置超过 15 cm；单回合最多 10 秒。

开局先让笔在重力下落稳。偶发的无接触初始化会重新采样（最多 20 次）；回合开始后发生的掉落会计入失败，不会重置后继续计分。

允许掌面支撑。这是掌内定向的第一阶段；指尖换指、抛转以及连续转圈是后续任务。绿色半透明笔只表示目标朝向，不参与碰撞。

## 复现

本次实验结果可在 [HTML 对比页](../pen.html) 查看。附带实验模型位于 `artifacts/pretrained/pen-v1-experimental/`，可直接载入推理。

```bash
# 加载附带的实验模型
.venv/bin/python scripts/pen_task.py evaluate \
  --checkpoint artifacts/pretrained/pen-v1-experimental/policy.zip \
  --video --output runs/pen-reproduction

# 训练；目录需为新目录
.venv/bin/python scripts/pen_task.py train \
  --steps 120000 --num-envs 4 --output runs/my-pen-ppo

# 独立评估，默认测试种子 20000–20031
.venv/bin/python scripts/pen_task.py evaluate \
  --checkpoint runs/my-pen-ppo/policy.zip --episodes 32 --video \
  --output runs/my-pen-evaluation

# 两个基线：保持初始命令、随机动作
.venv/bin/python scripts/pen_task.py evaluate --output runs/my-pen-hold
.venv/bin/python scripts/pen_task.py evaluate --random --output runs/my-pen-random

# 从相同任务的 checkpoint 续训，使用新输出目录
.venv/bin/python scripts/pen_task.py train \
  --resume runs/my-pen-ppo/policy.zip --steps 300000 \
  --output runs/my-pen-continued
```

每次训练保存 `config.json`、`policy.zip`、周期 checkpoint、逐回合日志和 `progress.json`。评估保存每回合结果、前三回合轨迹和可选视频；checkpoint 与配置必须一起保留。基础手势策略和转笔策略的观测、动作及任务不同，不能混用。

## 本次试训结果

每组使用相同的 32 个独立测试种子（20000–20031）：

- 不动手：成功 **0/32**，掉落 **0/32**，平均最终方向误差 **90.4°**。
- 随机动作：成功 **0/32**，掉落 **13/32**，平均最终方向误差 **93.3°**。
- PPO 首轮 120,832 步：成功 **0/32**，掉落 **0/32**，平均最终方向误差 **55.7°**。
- PPO 续训至约 60 万步：成功 **1/32**，掉落 **6/32**，平均最终方向误差 **43.9°**。

续训改善了方向误差，也增加了掉笔风险。它已在一个独立回合完成任务，但整体成功率仍低，尚未学会可靠的掌内定向。下一步应优先研究稳定抓持初始化和从小角度到大角度的训练课程，再挑战连续转圈。

HTML 保留固定顺序的前三回合视频，另标出唯一成功回合（seed 20011）。该回合单独重复推理仍然成功：7.5 秒完成，结束时方向误差约 2.58°。这个筛选示例不改变 32 回合的总体统计。

通过了 32 项项目测试，包括悬空对准不能成功、无动作不能误判成功和连续有效重置检查；另外检查了 1000 个初始化种子及 600 次连续重置。

## 官方实现与研究参考

优先复用 [ORCA 官方掌内方块任务](https://github.com/orcahand/orca_sim#sample-task-in-hand-cube-orientation) 的场景构造方式、掌心向上 v1 手模型和 MuJoCo 物理。新增训练笔、目标、观测与成功判定。上游文件未修改；生成的任务 XML 在 `local/tasks/v1/pen.xml`。

[PenSpin](https://penspin.github.io/) 研究了连续转笔，采用仿真教师策略、学生策略和少量实机数据微调。[其代码](https://github.com/HaozhiQi/penspin) 使用 Allegro 手，[安装环境](https://github.com/HaozhiQi/penspin/blob/main/docs/install.md) 包含 Isaac Gym 与 CUDA；模型无法直接用于 ORCA v1。这里参考任务分阶段的思路，当前实现使用本机 MuJoCo 和 SB3 PPO，没有移植或运行其预训练策略。

[HORA](https://haozhi.io/hora/) 提供了柱状物体旋转与自适应控制的研究参考，可用于后续的连续旋转和物体参数变化任务。

## 从这里到真机

1. 建立可靠初始抓持，扩大独立评估；逐步改变笔的质量、摩擦、尺寸和初始朝向。
2. 从 90° 定向发展为连续方向控制，增加换指，减少掌面支撑。
3. 辨识飞特舵机、腱绳和皮肤摩擦；加入延迟、传动误差与接触参数变化。
4. 获取真实笔的位置和方向，或训练只用可测信号的学生策略。现有 RGB 页面识别人手骨架，尚不跟踪笔，因此不能直接给这个教师策略提供完整观测。

本阶段训练和评估仅在仿真内运行。任务是否学会，以独立评估的成功率和掉落率为准，不能仅看训练奖励上升。
