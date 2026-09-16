# STS3215 真机控制

**状态：协议和异常路径已通过自动化测试，实体手尚未验证。** 接下来需要把每个舵机的实际运动与模型关节对应起来。

## 1. 检查连接

使用 USB 串口直通板、外部舵机电源与公共地线。关闭飞特调试软件，避免同时占用串口。

```bash
orca-hand doctor
orca-hand scan --port /dev/cu.usbserial-实际设备名 --baudrate 1000000
```

扫描只 ping 和读取模式，不启用舵机。波特率以实际配置为准；首版要求 STS **位置模式 0**，不会自动修改模式、ID、EEPROM 或零位。

## 2. 建立设备配置

```bash
mkdir -p local
cp configs/hardware.template.yaml local/hand.yaml
```

编辑 `port`。模板的 `motor_id` 依据装配教程整理，仍需逐关节核验，特别是 ID 13、14 与拇指根部关节的对应关系。

`thumb/index/middle/ring/pinky` 分别对应拇指到小指；`abd` 为侧摆，`mcp/pip` 等为模型中的屈伸关节。严格以模型中的轴和实际运动为准。

## 3. 逐关节标定

先在仿真查看该手指，再开始标定：

```bash
.venv/bin/mjpython -m orca_feetech.cli demo --poses open,index,open --output runs/check-index
orca-hand poses
orca-hand calibrate --config local/hand.yaml --joint index_mcp --output local/hand.yaml
```

工具显示候选 ID、实读型号码、模型中立角和范围。输入舵机 ID 确认后：

- `+` / `-`：低速移动 **5 个编码值**，观察实际关节与方向。
- `read`：读取位置、原始电流反馈和温度。
- `lower` / `upper`：在安全可用的端点输入对应模型角度并记录编码值。
- `neutral`：让实际关节与模型中立姿态对齐后记录编码值。
- `save`：检查三点单调性和范围并保存；`q` 或 Ctrl+C 卸力退出。

角度是 **v1 MJCF 模型坐标的度数**，不是舵机轴角。模型中立姿态含参考角，部分关节中立值不为零。请结合模型和实际关节角度量取锚点；仅凭舵机转角不能自动确定手指角度。

不要以撞到机械止点的方式寻找范围。空闲 10 秒会卸力，下次点动从实读位置重新启用。三点用于分段线性映射；更换腱绳、调节棘轮或线轮后应重新标定。

依次完成全部 17 个关节，工具才会把整手 `calibrated` 设为 `true`。不要手工把空配置标成已标定。

## 4. 动作与回放

```bash
orca-hand demo --backend hardware --config local/hand.yaml \
  --poses open,index,open --duration 4 --output runs/hardware-index
```

确认单指运动后再测试 `open,fist,pinch,open`。每次运行保存目标、下发值、估计关节位置和总线反馈。

同一轨迹先仿真再真机：

```bash
orca-hand replay runs/video-test/trajectory.jsonl --headless --video --output runs/replay-sim
orca-hand replay runs/video-test/trajectory.jsonl --backend hardware \
  --config local/hand.yaml --output runs/replay-hardware
```

回放从实读位置平滑过渡到轨迹起点，按记录时间重采样；真机不能使用 `--fast`。

## 运行行为

- 初始目标先设为当前编码值，再启用舵机，避免跳到上次保留的目标。
- 20 Hz 控制，目标速度上限 0.8 rad/s；预设动作使用五次多项式平滑。
- 输入过期 250 ms 后停止推进目标；2 秒无新输入触发独立看门狗卸力，需要重新启动。
- 读写异常、温度阈值或退出会尝试卸力；拔线后无法保证指令送达，保留未确认状态。
- 看门狗运行在主机进程内；强制杀进程或主机掉电时不能保证卸力，实体断电方式仍需可用。
- 仅使用位置控制。电流保留为原始值，不换算为接触力或真实闭环力矩。
- 反馈角度来自舵机与标定的换算，不能检测所有腱绳松弛、滑移和接触变形。

上机验收：17 路 ID／方向、三点与重复性、单指、整手、回放、遮挡后停止和卸力。请保留对应视频与运行目录，再更新“真机通过”状态。
