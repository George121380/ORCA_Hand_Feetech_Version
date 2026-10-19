# ORCA v1：可打印物体的掌内转向与 sim-to-real

这套新任务为 **ORCA Hand v1 右手、17 个 STS3215**设计，固定底座、掌心向上、手腕保持模型中立角。首先做方块或训练笔的掌内偏航定向，允许掌面支撑。它不是任意轴翻方块，也不是已经学会的连续花式转笔。

[交互设计网页](https://orca-v1-sim2real-lab.peiqi.chatgpt.site)展示 3D 几何、贴码位置与实验流程。网页源文件在 [../sim2real/](../sim2real/)，用 HTTP 服务打开，不能依赖 file:// 读取 JSON。所有旋转动画都是设计示意。

新版任务尚无训练成功率或真机成功率。Mac 只做代码、几何、合成视觉和仿真功能检查；长训练由你的 Linux 4090 执行。旧的 pen-v1-experimental checkpoint 与此任务不兼容。

## 1. 几何、打印与两端标记

当前默认几何为 **printables-v4**：55mm方块；**210mm总长、Ø16×146mm圆杆、24×24×22mm方头**的训练笔。170mm旧笔是按一体竖直打印约束选的紧凑版本，未证明适合操作；现按手部几何重新评估，将210mm作为首选候选，不宣称最优。v3实体、JSON和checkpoint不能混用；旧文件保留在assets/objects/v3，供追溯。纸质标记图案与黑框尺寸不变，已打印的贴纸可复用，但须按新位置粘贴。

### 尺寸选择的证据与限制

[可复现脚本](../../scripts/assess_pen_size.py)比较170/190/200/210/220/230mm，只增加圆杆长度，保持直径、方头与肩部不变。固定4种模型手型、3个高度、37个偏航角，对4个物体中心X位置分别做444张静态相交快照；完整数据见[JSON](../sim2real/pen-size-review.json)。这些相交快照不是动作轨迹、碰撞概率、力闭合或训练成功率。

在X=190mm区域，方头与手指相交快照为170版170/444、200版72/444、210版31/444、220版29/444。210相对170增加40mm操作圆杆；210到220在该区域收益较小，而均匀实体近似下偏航惯量从1.74倍170版升至1.96倍（再增约13%）。完整数据也保留其他位置，不能把这个局部结果泛化为全部工作空间。每个长度另从前移位置做40次独立重力落稳检查，均有手部支撑、无底座接触；仅证明初始化可行。

另以示例针孔相机检查标签几何可见性：相机位置[.25,-.10,.48]m、目标[.185,.05,.19]m，1920×1080、焦距1400px，要求短边≥40px、正面、四角在画面且射线无遮挡。X=190mm快照中，170版两端均有完整可见标签为382/444，210版427/444，220版428/444。它没有模拟真实图像检测、模糊、曝光与定位噪声，不能当检测准确率或替代真实标定。

初始化笔的投放X从135mm前移到190mm，实际重力落稳后可向掌面回落；不再把底座支撑计为有效手内接触。长笔在部分转向姿态仍可碰底座，策略须主动移位/抬升，最后仍用独立训练与实物抓持验证。200/220mm保留为后续比较候选；当前训练入口固定使用210mm并记录几何签名。

### 直接打印

下载[完整v4打印包](../../assets/objects/v4/a1mini-print-pack.zip)。Bambu Studio打开 **pen-split-a1mini-pla.3mf**：一片210×24×12mm半壳，沿床面45°，另有3根定位销。**同工程完整打印两次**，获得两片同形半壳、3根使用销和3根备用销。不自动重排、不缩放。平面包络165.46×165.46mm，计入3mm brim与0.1mm间隙后的最小床边余量约4.17mm。无需外部支撑，盲孔顶有4.3mm小桥接。

本机Bambu Studio 2.5 / A1 mini 0.4 / Generic PLA已切片：每盘60层、高12mm、用料21.18g、模型1h18m、总计约1h26m；两盘约2h52m。用料含brim与备用销，不是成品称重。方块打开cube-a1mini-pla.3mf，55mm立方体平放；之前同几何工艺切片为275层、60.13g、模型3h54m，总计4h06m。都尚未实打。

默认工艺：0.2mm层高，半壳4层壁、方块3层壁；20% gyroid，顶底各5层，外墙40、内墙60mm/s，默认加速度1000mm/s²。半壳brim3mm、象脚补偿0.15mm；方块brim8mm。填充比例不是成品密度比例，外壁、接缝底面和孔壁都会加料；不要临时填实或加配重后仍沿用原物理参数。[Bambu Studio官方参数定义](https://github.com/bambulab/BambuStudio/blob/master/src/libslic3r/PrintConfig.cpp)对应预设字段。

一体210mm笔也能床面对角放下，但两头方块着床时，146mm中杆底部悬空4mm，支撑会影响滚动表面；空间倾斜仍增加悬垂，因此不作为默认工程。pen-assembly-reference.3mf、pen.stl、pen.scad是完整装配外形参考。pen-split-a1mini.3mf为已摆放的标准毫米几何；pen-half.stl需打印两片并自行转45°、保持接缝平面贴床，pen-alignment-pin.stl总共需三根。PETG备用工程仅在换用相应材料时选，当前确认材料为PLA。

再生成（开发机安装sim、vision、print extras；训练机不需要打印依赖）：

    uv sync --extra sim --extra vision --extra print
    python scripts/manipulation.py export-objects --output assets/objects/v4
    python scripts/prepare_a1mini.py --output assets/objects/v4
    python scripts/prepare_print_kit.py --output assets/objects/v4
    python scripts/assess_pen_size.py
    python scripts/render_object_scale.py
    python scripts/package_print_assets.py

prepare_a1mini.py需要本机Bambu Studio官方预设。4090安装脚本之后不要直接uv sync，以免恢复CPU torch；见下文。

### 实体、标签与装配

同一objects.py生成毫米STL、OpenSCAD、米制仿真凸体、AprilTag角点及设计尺寸。整笔为非凸形状，以两个方头、两个肩部和一个64边圆柱精确分解，避免整体凸包填粗中杆。分体半壳的外部并集经布尔比较与装配外形一致，只有内部定位孔间隙；纸厚、胶层与制造误差仍需实测。

- 方块55×55×55mm：六面黑框32mm、白纸40mm、四边余7.5mm，ID0..5。ID0/1在±X，2/3在±Y，4/5在±Z。
- 笔总长210mm，纯圆杆Ø16×146mm，两端各10mm渐变肩部与24×24×22mm方头。
- 方头侧码中心Y=±94mm，基线188mm；四侧面+X/+Z/-X/-Z，负端ID10/12/14/16，正端ID11/13/15/17。黑框16mm、白纸20mm，纸片居中，宽向各余2mm、轴向各余1mm，顶边统一朝+Y。
- 外端面ID18位于Y=-105mm、ID19位于+105mm，黑框/纸同为16/20mm，顶边朝+Z；圆杆和肩部不贴码。
- 固定参考码ID40，黑框50mm、纸66mm，固定在独立刚性平板上，放在手旁。中心为原点，纸面右为+X、上为+Y、法线向外为+Z。

[A4手册PDF](../../assets/objects/v4/print-kit-a4.pdf)六页含标签、校验尺、20mm棋盘格（9×6内角点）、尺寸与ID/方向。按实际大小100%，不适应/缩小页面；第4页横向。用尺复核100mm标尺与黑框边长，保留白边，裁掉说明文字与箭头。哑光不透底白纸/贴纸，不覆亮面胶。PnP边长是黑框边长，不含白边。[OpenCV官方说明](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)介绍字典与角点检测。

定位销Ø4×6mm，盲孔Ø4.3、单边深3.15mm，Y=-94/0/+94mm。先试装、去毛边，不能强压；接缝薄涂适用于PLA的胶，夹紧至固化，不在圆杆、方头外表面留下胶或台阶。公差与胶接强度尚未实打验证。

称量粘好、去brim、贴码的成品，用--mass-kg输入；名义55g方块、40.2g笔只是初值。惯量仍按均匀实体近似，20%填充、壁厚与接缝底面造成的分布误差需辨识与随机化，不能称为已经完全匹配的数字孪生。

设计参考[WUJI官方圆杆加方头方案](https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/pen_spin/docs/setup.md)，但尺寸、标准AprilTag字典和手部接口独立设计，未直接复用它的物体或策略。

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

### 后台执行一个完整实验

`scripts/run_training_job.py`把一段训练、checkpoint选择、正式测试、基线和视频串在一起，限Linux执行。当前210mm笔可先用低随机化建立15°技能：

    tmux new -s orca-pen
    .venv/bin/python scripts/run_training_job.py --object pen --angle 15 \
      --randomization .3 --steps 3000000 --num-envs 16 --device cuda \
      --video --output runs/pen-15-r03-seed0

用Ctrl+B、D离开tmux，SSH断开后任务继续；`tmux attach -t orca-pen`查看，Ctrl+C中断当前任务。不要重复使用已存在的输出目录。模型、TensorBoard和逐环境monitor位于`model/`，训练stdout在`train.log`，阶段状态、实际命令和子进程PID在`job.json`。可以另开SSH查看：

    tail -f runs/pen-15-r03-seed0/train.log
    cat runs/pen-15-r03-seed0/model/progress.json

训练期间内部评估仍使用10000+seed。结束后，仅在开发集30000..30039上比较最终policy与内部best_model，按成功率优先、掉落率其次、最终误差再次选择，复制到`selected/`并保存对应config。随后只对选中模型运行20000..20099正式测试，静止和随机动作基线使用相同初始种子；这些正式测试不参与checkpoint选择。两个种子区间不能重叠，多种子实验应显式规划独立区间。`test-policy/metrics.json`给出100回合结果和区间；`video/evaluation.mp4`记录正式测试固定前五回合，不筛选成功片段。

`job.json`的`status=complete`只表示命令执行完毕，`acceptance_met`才表示该阶段达到80%成功、至多10%掉落的初步门槛。即便低随机化15°达标，也不能当作完整90°转笔或sim-to-real完成；应提高随机化后重新训练、验证，再推进角度。成品质量和真机ROM未测量的实验必须标记为名义物理参数实验。若根据正式测试结果改任务或调参，应另留新的最终测试种子，不能反复把同一组种子称为未见测试集。

需要无人值守继续课程时，可在另一tmux会话执行：

    .venv/bin/python scripts/run_pen_curriculum.py \
      --initial-job runs/pen-15-r03-seed0 --output runs/pen-course-seed0

它先等待首段结束，只有该段达标才依次续训15°/随机化1、45°/1、90°/1（各300万步），最后连续四次90°/1（1000万步、40秒回合）。含首段总预算2200万步。各段正式测试种子依次21000、22000、23000、24000，开发集依次31000..34000，避免把已经用于推进课程的测试回合重复当作最终未见测试。任一阶段未达门槛立即停止并保存`quality_gate_failed`，供检查失败视频和接触情况，不自动假定它已经学会。首段配置与模型文件必须保留，后续阶段沿用相同质量、摩擦和关节限位合同。`curriculum.json`记录阶段与最后选中checkpoint。

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

成品质量称量后换算成kg，再用 --mass-kg 输入，摩擦用 --friction；执行延迟范围可设 --delay-max-ms 0/50/100/150/200，目标偏差 --command-bias-deg，关节估计偏差 --encoder-bias-deg。默认范围只是起点。续训必须保持物体、质量、摩擦、动作速度和 ROM 合同一致；课程角度、回合长度、目标次数与随机化程度允许变化。

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

当前任务ID为`orca-v1-printable-yaw-v5`，打印几何仍为`printables-v4`。每帧58维：16关节角（编码器映射估计）、16上一位置目标、3物体在固定手部仿真世界S系的位置（除以0.2m归一化）、3物体相对初始位置、9物体旋转矩阵、9目标旋转矩阵、有效标志、观测年龄。四帧堆叠共232维。没有物体真值速度、接触标志或无法在真机获得的关节速度输入。仿真真值只用于生成带噪视觉信号、奖励和验收。

Jarvis首轮v4观测仅包含相对初始位置；40个初始化种子中，笔落稳后的X位置跨度约38.7mm，这会隐藏影响手指接触的初始位置。v5加入已标定相机本来就能提供的S系绝对位置，并在仿真、真机共用的ObservationCodec里统一处理。旧220维checkpoint属于对照实验，不能载入232维任务续训或部署；任务ID与维数校验会拒绝混用。这里没有因此声称精度或成功率已经提高，需要新一轮留出评估。

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
