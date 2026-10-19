# ORCA v1：可打印物体的掌内转向与 sim-to-real

这套新任务为 **ORCA Hand v1 右手、17 个 STS3215**设计，固定底座、掌心向上、手腕保持模型中立角。首先做方块或训练笔的掌内偏航定向，允许掌面支撑。它不是任意轴翻方块，也不是已经学会的连续花式转笔。

[交互设计网页](https://orca-v1-sim2real-lab.peiqi.chatgpt.site)展示 3D 几何、贴码位置与实验流程。网页源文件在 [../sim2real/](../sim2real/)，用 HTTP 服务打开，不能依赖 file:// 读取 JSON。所有旋转动画都是设计示意。

新版任务尚无训练成功率或真机成功率。Mac 只做代码、几何、合成视觉和仿真功能检查；长训练由你的 Linux 4090 执行。旧的 pen-v1-experimental checkpoint 与此任务不兼容。

## 1. 几何、打印与两端标记

优先下载 assets/objects/v2/ 中的 pen-a1mini-pla.3mf 或 pen-a1mini-petg.3mf 原生工程（方块同理），按实际耗材选择。它们含 A1 mini 0.4mm、纹理 PEI 和工艺配置，不含已发送的打印任务。标准 *-a1mini.3mf 只包含毫米几何，*-a1mini.stl 为已摆正方向的备用网格。模型尺寸都是100%，不得为了放入打印空间再次缩放。PLA 笔工程已在本机 Bambu Studio 2.5 成功切片（850 层、170mm、无支撑、估计27.74g、模型打印2h31m）；没有进行真实打印验证。PETG 工程按对应官方耗材预设导出，仍需按实际耗材复核切片。

运行：

    python scripts/manipulation.py export-objects --output assets/objects/v2

同一个 objects.py 生成毫米 STL、可编辑 OpenSCAD、米制仿真凸体与 AprilTag 角点。STL 本身没有单位，切片软件必须按 **mm / 100%** 导入。白色贴纸区是现有的平面，不是凹槽，也没有额外端头配重。碰撞形状与裸打印件相同；纸和胶厚度不含在模型中，应保持薄且一致，打印尺寸误差和皮肤接触差异仍需实测。

- 方块：42 × 42 × 42 mm。六面黑框 24 mm、白纸 30 mm，ID 0–5。
- 默认训练笔改为圆柱中杆 + 方形端头：总长 170 mm，中间 Ø16 mm ×106 mm；两端各 10 mm 渐变肩部，再接 24×24×22 mm 方头。
- 每个方头四个侧面贴码，ID 10–17，黑框 16 mm、白纸 20 mm；侧码中心 Y=±74 mm，两端基线 148 mm。
- 外端面 ID 18 位于 Y=-85 mm、ID 19 位于 Y=+85 mm，标记尺寸也为 16/20 mm；中间圆杆完全不贴码。
- 静态参考板：ID 40，黑框 50 mm，白纸 66 mm。牢固放在底座旁，与物体同时入镜。

方块中心为 O 原点：ID 0/1 在 +X/-X，2/3 在 +Y/-Y，4/5 在 +Z/-Z。每枚标签的中心、右方向 u、上方向 v 与角点都在对应 JSON 中；角点顺序为解码后的左上、右上、右下、左下。标记外法向为 u×v。

笔的 ID 18/19 定义负/正 Y 端。方头侧面 f=0/1/2/3 对应 +X/+Z/-X/-Z，负端标签 ID=10+2f，正端 ID=11+2f。侧码顶边朝 +Y，端面码顶边朝 +Z。不要交换或旋转标签。

默认改型参考了 [WUJI 官方装配资料](https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/pen_spin/docs/setup.md) 中的圆柱握持段与方形端头思路。它更接近最终转笔接触，且能容纳更大角点，但需要明确建模端头碰撞和惯量。原六棱方案适合较早的定向调试，滚动接触与圆柱不同；这次不是以它作为最终打印件。WUJI 的尺寸、定制标签字典与手的控制接口没有直接复用，本项目使用自己的标准 AprilTag 36h11。

特别注意：整支笔不是凸形状。仿真以两块方头、两块渐变肩部、一个 64 边圆柱组成五个不重叠凸体；打印外表面取相同实体的外部并集。没有用整个物体的凸包把细圆杆填成粗方杆。均匀密度的分块体积用于分配质量与惯量。


标签图纸为 [tags-a4.svg](../../assets/objects/v2/tags-a4.svg)。按 A4、100%、关闭适应页面打印，验证底部 100 mm 标尺，再检查黑框边长。纸张静区一起剪下，图纸上的文字和裁切线不贴进码区。使用哑光纸，避免亮面胶带。**位姿求解的边长指黑框外边长，不含白边**。[OpenCV 文档](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)说明了角点顺序与字典标记的生成。

A1 mini 的官方空间为 180×180×180 mm。已导出笔竖直放置、底部 Z=0 的 170 mm 一体件，以及平放的方块。建议原装 0.4 mm 喷嘴，0.2 mm 层高、3 层壁、20% gyroid、8 mm 外侧 brim、外墙 40mm/s、内墙 60mm/s、默认加速度 1000mm/s²。10mm 肩部使最大径向增长约9mm，竖直打印的设计悬垂不超过45°，首先在切片预览检查无支撑方案。高件仍需清洁热床、足够冷却与首层检查。用卡尺检查尺寸，称量贴好标记的成品。35 g 方块、24 g 笔只是名义初值，实际质量用 --mass-kg 输入；惯量默认按均匀实体外形分配，非均匀填充或配重需要额外辨识。不要在内部放未建模的配重。

## 2. 4090 安装与首次检查

建议 Ubuntu 22.04/24.04 x86_64、Python 3.11、可工作的 NVIDIA 驱动、16 个左右可用 CPU 核心。相机和舵机不需要连接训练机。

    git clone https://github.com/George121380/ORCA_Hand_Feetech_Version.git
    cd ORCA_Hand_Feetech_Version
    bash scripts/setup_4090.sh
    source .venv/bin/activate
    pytest -q tests/test_manipulation.py

安装先使用 uv.lock 和已固定的官方模型快照，再将 Linux CPU torch 替换为 PyTorch 2.8.0 CUDA 12.8 wheel，并实际分配 GPU 张量自检；其来源为 [PyTorch 官方安装页](https://pytorch.org/get-started/previous-versions/#v280)。没有驱动或 CUDA 不可用就报错，不会悄悄退回 CPU。

**GPU 环境随后不要直接执行 uv sync**，它会按仓库原有 CPU CI 配置恢复 CPU torch。更新代码后重新运行 setup_4090.sh。完整安装版本保存在 local/training-environment.txt，长训练记录还会保存 Python 平台、Torch、CUDA 与 Git revision。原生 MuJoCo 用 CPU 多进程模拟，CUDA 只加速 PPO 网络更新；[SB3 官方文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)也提醒小型 MLP PPO 往往更适合 CPU。可分别短测 --device cpu/cuda 后比较 progress.json 中的吞吐。4090 不等于全部物理计算都在 GPU 上。真正 GPU 物理的 MJX 移植是后续优化，目前没有声称完成。

首次短程自检（这不是任务学会的证据）：

    python scripts/manipulation.py train --object cube --steps 1024 \
      --num-envs 2 --device cuda --output runs/smoke-cube
    python scripts/manipulation.py evaluate --checkpoint runs/smoke-cube/policy.zip \
      --episodes 3 --output runs/smoke-cube-eval

缺少 EGL 时先不加 --video；无窗口训练不需要 OpenGL。Linux 视频渲染建议设置 MUJOCO_GL=egl 并安装相应 NVIDIA/EGL 运行库。不要在 Mac 上启动长训练。

## 3. 分阶段训练、续训与验收

建议先方块 15°，再 45°，再 90°，最后四次同向 90°；训练笔独立重新开始，不复用方块权重作为已学会的模型。

    python scripts/manipulation.py train --object cube --angle 15 \
      --steps 3000000 --num-envs 16 --device cuda --seed 0 \
      --output runs/cube-15
    python scripts/manipulation.py evaluate --checkpoint runs/cube-15/policy.zip \
      --episodes 100 --seed 20000 --video --output runs/cube-15-eval
    python scripts/manipulation.py evaluate --checkpoint runs/cube-15/policy.zip \
      --baseline hold --episodes 100 --output runs/cube-15-hold
    python scripts/manipulation.py evaluate --checkpoint runs/cube-15/policy.zip \
      --baseline random --episodes 100 --output runs/cube-15-random

若你已完成硬件标定，把 configs/hardware.local.yaml 从 Mac 复制到训练机，并在**第一次训练以及每次续训**中加入 --device-config configs/hardware.local.yaml，保证训练动作域就是实测 ROM。端口只用于解析配置，此时不会打开串口。未提供时按模型全 ROM 训练；若它超出真机实测范围，部署会拒绝，需要按实测范围重新训练。这一步最好在昂贵训练前完成。

成品质量用 --mass-kg 0.035（示例值），摩擦用 --friction；执行延迟范围可设 --delay-max-ms 0/50/100/150/200，目标偏差 --command-bias-deg，关节估计偏差 --encoder-bias-deg。默认范围只是起点。续训必须保持物体、质量、摩擦、动作速度和 ROM 合同一致；课程角度、回合长度、目标次数与随机化程度允许变化。

    python scripts/manipulation.py train --object cube --angle 45 \
      --resume runs/cube-15/policy.zip --steps 3000000 --num-envs 16 \
      --device cuda --output runs/cube-45
    # 通过留出集后，同样方式续训 90°。
    python scripts/manipulation.py train --object cube --angle 90 --turns 4 --horizon 800 \
      --resume runs/cube-90/policy.zip --steps 10000000 --num-envs 16 \
      --device cuda --output runs/cube-chain

笔任务使用 --object pen 并从新目录、新策略开始。所有步数只是初始预算，不保证收敛。randomization=1 为默认；可先用 --randomization 0.3 建立技能，再在同角度逐步升至 1。只有训练奖励上升、或某个成功视频，都不足以验收。

每段目标要求完整旋转误差 <8°、物体中心偏移 <4 cm、手部实际接触、线速度 <0.08 m/s、角速度 <1 rad/s，连续 0.5 秒。这个目标旋转由初始姿态绕仿真世界 Z 轴转动得到；笔的轴向翻转与滚转不能借物体对称性作弊。掉落定义为中心低于 10 cm 或离初始位置 >12 cm。多段模式每段完成后才更新目标，四段不中断，仍会检查掉落。

首次建议验收门槛：100 个从 20000 开始的留出回合，成功率至少 80%、掉落率至多 10%；metrics.json 保存 95% Wilson 区间和逐回合结果。训练内部评估用不同的 10000+seed 流；正式验收集不用于选最佳 checkpoint。前五个固定顺序回合录视频，不筛成功回合。后续应使用 0/1/2 三个训练种子及额外实测压力条件验证；当前没有这些训练结果。

训练保存 config.json、policy.zip、best_model.zip、checkpoint_*、monitor.csv、TensorBoard、progress.json；中断会尝试保存 interrupted.zip。checkpoint 与同目录 config.json 必须一起搬回 Mac。Task ID、几何、关节顺序、模型文件和上游 lock 都会校验，旧模型或改错尺寸会拒绝载入。

## 4. 观测、执行与随机化

20 Hz 控制，2 ms 物理步长。16 个手指动作是目标角速度，经 0.8 rad/s 速度、2 rad/s² 加速度限制后积分为位置目标；手腕固定模型中立角。仿真和真机共用 ActionCodec。

每帧 55 维：16 关节角（编码器映射估计）、16 上一位置目标、3 物体相对初始位置、9 物体旋转矩阵、9 目标旋转矩阵、有效标志、观测年龄。四帧堆叠共 220 维。没有物体真值速度、接触标志或无法在真机获得的关节速度输入。仿真真值只用于生成带噪视觉信号、奖励和验收。

奖励使用朝向进展、有限的运行成本与完成目标奖励，避免正的逐帧奖励鼓励一直等待；掉落惩罚随回合长度设定。随机化覆盖物体质量/惯量同比例变化、物体滑动摩擦、舵机位置增益、关节阻尼、0–100 ms 默认执行延迟、目标偏差、编码器估计偏差、视觉位置/偏航噪声、采样保持和短时丢帧。它没有完整模拟腱绳、皮肤形变、相机曝光和每个可见面遮挡，因此 domain randomization 不能替代实物测量。丢帧时两端都使用相同的停止推进规则，位姿年龄 >250 ms 会结束 rollout。

## 5. 相机与固定参考系

准备固定相机，至少 1080p，斜上方同时看到两端尽可能多的码和参考码 40。锁定焦距/对焦，短曝光、柔和照明。方块码比笔码更容易检测；当前最短可见黑框边需至少 40 px（可根据实拍再调整）。在 fx≈1400px、距离 28 cm 时，正视 16 mm 侧码约 80px，但透视斜视会进一步变小。过远或码接近侧视时不能保证可用。

使用单相机中的所有有效物体标签角点，在共同物体坐标里做联合 PnP；两端同时可见提供更长基线。不是分别估计两个位姿再平均。单端保留降级追踪；较大的平面歧义、过小角点、重投影 RMS >2.5px 或短时位姿跳变会被拒绝，不能因为有二维码就承诺毫米级准确。多相机融合未实现。

拍摄至少 15 张不同位置、倾角的棋盘格照片（默认 9×6 **内角点**、格长实测 20mm），必须使用部署分辨率、不缩放、不改焦距：

    python scripts/calibrate_vision.py intrinsics \
      --images 'local/calibration/*.png' --cols 9 --rows 6 --square-m .02 \
      --output local/camera.json

程序计算逐视图误差，RMS >0.8px 拒绝。标定 RMS 不是物体位姿精度；另做静态卡尺距离、已知角度与遮挡测试。

C 是 OpenCV 相机系，W 是参考板 40 的坐标系（右为 +X、上为 +Y、出纸面为 +Z），S 是仿真世界系，O 是物体坐标系。检测输出 T_CO 与 T_CW；真实输入使用：

    T_SO = T_SW @ inverse(T_CW) @ T_CO

参考板与手底座都固定。T_SW 必须实测登记，不能把单位阵当作已标定。选至少三个非共线、跨度 >1cm 的底座/CAD 可重复地标，测它们在参考板坐标中的米制 3D 坐标，并从官方模型获得相同点的仿真坐标，按同样顺序写 JSON：

    {"world_points_m": [[...], [...], [...]],
     "simulation_points_m": [[...], [...], [...]]}

用四个以上地标保留冗余更好。不要用会动的指尖作安装标定地标。参考板平面坐标只适用于平面上的点，其余点还需测高度。刚体配准残差 >3mm 拒绝，但小残差并不证明坐标对应正确，仍需用留出地标检查。

    python scripts/calibrate_vision.py align-world --points local/landmarks.json \
      --output local/alignment.json

移动相机不改变刚性 W/S 关系，但必须重新验证内参与视角；移动参考板或底座则必须重新登记 T_SW。每次控制都同时要求物体码和参考码，缺参考码也按丢观测处理。未来可以在固定相机下缓存外参，但当前采取这一保守策略。

## 6. 从 dry-run 到真机

现有 [硬件标定流程](hardware.md) 和 src/orca_feetech/calibration.py、hardware.py 继续使用；STS 接口、逐关节三点标定、速度限制、温度门槛与独立 watchdog 已存在。这里没有在你的实物上验证映射，也未确认另一个独立图形版调试器的位置。标定 YAML 的 verified 必须由真实检查填写，不能批量改成 true。

可先用只读工具记录手动运动的编码器，无写命令或使能转矩；不要让两个程序同时占用串口：

    python scripts/identify_actuators.py record-passive \
      --device-config configs/hardware.local.yaml --seconds 10 --output local/passive.jsonl

现有动作/回放记录采用 trajectory.jsonl 的 actual/command 字段。闭环部署记录采用 q/command/execute 字段；identify_actuators.py fit 目前读取后者，可估计编码器响应延迟与偏差：

    python scripts/identify_actuators.py fit --log local/executed.jsonl \
      --output local/actuator-fit.json

这不等于腱绳传动辨识。要分别在正反向慢动作、不同腱绳张力下测真实关节角，测回差；用倾斜板法粗测物体与手皮肤静摩擦。若实际延迟/误差超出训练范围，修改范围并重新训练。先做逐关节空载低速动作，确认手腕的中立角和掌心姿态与仿真相同。

把通过仿真验收的 checkpoint 与 config.json 拷回 Mac，安装 --extra train --extra vision --extra hardware 以执行 CPU 推理。程序默认只做虚拟关节反馈的 dry-run，不打开舵机串口：

    python scripts/deploy_manipulation.py \
      --checkpoint runs/cube-15/policy.zip --camera-config local/camera.json \
      --alignment local/alignment.json --source 0 --seconds 10 --log local/dryrun.jsonl

dry-run 用来检查视觉、坐标、观测和命令范围；虚拟关节反馈不是实物响应，也不构成 sim-to-real 成功。检测必须在 10s 内建立，物体需处于训练掌区。通过后，在固定底座、手动放置物体且已完成标定的条件下短时执行：

    python scripts/deploy_manipulation.py \
      --checkpoint runs/cube-15/policy.zip --camera-config local/camera.json \
      --alignment local/alignment.json --source 0 --seconds 3 --log local/execute-01.jsonl \
      --device-config configs/hardware.local.yaml --execute

执行会先校验训练 ROM 是实测 ROM 的子集，初始手姿接近中立，再使用既有 HardwareBackend。后台相机线程不能阻塞控制循环；短时丢码保持命令，超过 250ms、控制落后 >100ms 或物体离开保持区域会结束并关闭后端，后端已有独立 2s 输入 watchdog。退出时禁用转矩会使物体下落，所以使用接物托盘。硬件断线无法保证收到禁转矩命令，保留物理断电手段。

真机只能报告视觉稳定目标，不冒充仿真的接触成功判定。真机验收另外人工标注是否留在手中、是否掉落，报告实际相机延迟、成功率和故障记录。在这一步完成前，不应声称策略已成功迁移。

## 7. 当前验证范围

新增测试覆盖闭合且外法向一致的可打印网格、所有贴纸的四角不超出平面、笔两端标记基线、两个物体的接触初始化、静止策略不能白拿成功、悬空对准不能成功、同一动作编解码的加速度/丢码保持、合成印刷 AprilTag 的实际检测、两端联合角点位姿与单端降级、噪声下长基线对笔轴方向的帮助。

这些检查验证实现，不证明策略收敛、打印公差、真实视觉精度或真实操控效果。Linux CI 另跑短程新任务训练与重载评估；4090 CUDA 安装/吞吐与真实手均需在对应机器完成。

研究参考：[ORCA 官方仿真](https://github.com/orcahand/orca_sim)、[PenSpin](https://penspin.github.io/)和 [HORA](https://haozhi.io/hora/)。当前直接训练可部署观测的策略，没有声称移植它们的教师学生或在线适应算法。


打印工程可用 python scripts/prepare_a1mini.py 重新生成，需要本机已安装 Bambu Studio。它从已清理个人元信息的原生工程 seed 和官方安装目录中的配置读取，不发送打印任务。
