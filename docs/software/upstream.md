# 官方实现与版本

调研时间：2026-09-14。完整提交号见根目录 `upstream.lock.json`，Python 版本见 `uv.lock`。

## 主流程采用的项目

- [`orca_core` · 4a99afd8ef8b](https://github.com/orcahand/orca_core/tree/4a99afd8ef8b368d4cb84a10e4a1d5e732aca660)：复用 `BaseHand`、`OrcaJointPositions` 和重定向配置读取。硬件自动电流／拉紧流程不适用于本项目的 STS 路径。
- [`orca_sim` · 05e00dafd0a6](https://github.com/orcahand/orca_sim/tree/05e00dafd0a6eb2186d69d4e351c252a7bcffadd)：复用 v1 MuJoCo 环境和全部物理参数，增加任务奖励、目标、训练和评估。
- [`orcahand_description` · b9b349a21ee0](https://github.com/orcahand/orcahand_description/tree/b9b349a21ee0238c62b6cf92ae7597027867adf8)：复用 v1 URDF/MJCF，用于官方 IK 求解和跨格式运动学检查。
- [`orca_teleop` · 062ddeb8c2b1](https://github.com/orcahand/orca_teleop/tree/062ddeb8c2b182d5c9cd8bd406cb07a532c97fc8)：复用 MediaPipe 模型资产、关键点约定和 `adaptive_analytical` 求解器。算法内部使用 Pinocchio、NLopt、尺度估计和低通滤波。
- [`FTServo_Python`](https://github.com/ftservo/FTServo_Python/tree/a203373036723e0d98c6c49b67cf09a9ee299220)：按飞特 STS 协议控制，安装 `ftservo-python-sdk==2.0.0`。
- [`Stable-Baselines3`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)：采用 PPO 训练实现。ORCA 基础环境不包含本项目的手势跟踪训练任务。

## 必要适配

**STS 与 HLS 的报文差异。** ORCA 快照中的 `sms_sts.py` 在 44–45 字节填入力矩；飞特 SDK 将 STS 的这两字节定义为运行时间，位置命令置零，HLS 才在这里填入力矩。本项目直接使用 SDK 的四参数 `SyncWritePosEx`，测试检查完整的七字节负载。原始电流反馈不映射为力矩或接触力。

**v1 的命名和参考角。** 官方重定向 YAML 的 `thumb_pip → thumb_cmc` 别名与 v1 不匹配。项目生成独立 v1 配置，清空该别名，按 MJCF 的范围与参考角设置输入／输出坐标。原始官方文件不改动。

**v1 的指尖与 DIP 关键点。** 锁定 URDF 的 `*_fingertip` 已经包含末节长度，官方求解器又叠加 30–45 mm 偏移。项目在求解器实例上清除额外偏移，再重新建立尺度估计参数。四根非拇指只有两个屈伸关节，默认 PIP / DIP 帧又恰好重合；项目把末节中点作为虚拟 DIP，保留真实 PIP 和指尖位置。这是人体三段手指到机器人两段手指的几何近似，不增加机械自由度。独立 URDF 正运动学测试覆盖这些关键点；没有修改官方优化目标或源文件。

官方 `OrcaHand` 构造会输出未标定舵机警告，重定向时其实只读取运动学配置。项目仅在求解器构造期间捕获这段 stdout，异常仍会传出，真实硬件错误不隐藏。

## 已审查的其他官方仓库

- [`orca_retargeter`](https://github.com/orcahand/orca_retargeter)：旧的独立重定向器；本项目选择功能更完整的 `orca_teleop`。
- [`faive_gym_oss`](https://github.com/orcahand/faive_gym_oss)：Faive Hand 与 Isaac Gym Preview 4 的 RL 框架，可参考掌内操作任务设计，但不是本机 v1 模型的直接运行入口。
- [`orca-gym`](https://github.com/orcahand/orca-gym)：当前默认分支仍是 K-Sim 人形行走模板；不能据仓库名称推断已提供 ORCA 手势训练。
- [`srl_il`](https://github.com/orcahand/srl_il)：通用模仿学习框架；适合有示教数据后的后续阶段。
- [`rwr_system`](https://github.com/orcahand/rwr_system)：课程时期的遥操作、数据采集与推理系统，列为参考。

## 第三方资产

官方快照在 `.upstream/` 中保留原始来源与许可证文件，不整体复制进本仓库。生成的模型元数据和配置源自上述 ORCA 文件。飞特 SDK 的来源与许可证随安装包保留。

视频测试图片来自 Google MediaPipe 的测试资产，保存在 Git 忽略的 `local/fixtures/`；脚本记录来源与 SHA-256。文档中的相关演示明确标成静态图片功能测试，不作为真人实时遥操作证据。

升级上游时先更新锁定提交，再运行命名、参考角、报文与回归测试；不要把 `main` 当成稳定的运行版本。
