# 环境安装

首版使用 Python 3.11、Mac 本地摄像头与 USB 舵机。Linux 运行相同代码；PPO 小网络默认 CPU，Linux 的 PyTorch 从官方 CPU wheel 源安装，避免额外下载 CUDA。

```bash
python3 scripts/bootstrap.py
uv sync --frozen --extra hardware --extra train --extra teleop
source .venv/bin/activate
orca-hand doctor
```

bootstrap 根据 `upstream.lock.json` 获取四个官方仓库的固定提交，放入 `.upstream/`。`uv.lock` 固定 Python 依赖。来源标记不匹配时，先把该快照目录移到别处再重新获取；工具不会覆盖已有目录。

## 按用途安装

```bash
uv sync --frozen --extra hardware       # 总线、标定和控制
uv sync --frozen --extra sim            # 仿真与视频导出
uv sync --frozen --extra train          # 仿真与 PPO
uv sync --frozen --extra teleop         # 视频与仿真重定向
```

希望功能共存时，把需要的 `--extra` 放在同一次命令中。安装后直接使用 `.venv/bin/orca-hand` 或激活环境，避免 `uv run` 无意重新同步可选依赖。

## Mac

- 交互式 MuJoCo 必须用 `.venv/bin/mjpython -m orca_feetech.cli ...`。
- `--headless --video` 可直接使用 `orca-hand`。
- 首次使用摄像头时，在 macOS 设置中允许当前终端／应用访问摄像头。
- 串口通常形如 `/dev/cu.usbserial-*`。`doctor` 列出端口；蓝牙和 debug-console 不是舵机。

## Linux

Ubuntu 22.04 是 CI 的目标系统；运行硬件需有串口权限，例如加入 `dialout` 组后重新登录。

```bash
orca-hand train --num-envs 8 --steps 300000 --output runs/linux-policy
```

纯训练和无视频评估不创建 OpenGL 上下文。有 NVIDIA 驱动的无桌面机器导出视频可使用：

```bash
MUJOCO_GL=egl orca-hand infer \
  --checkpoint artifacts/pretrained/gesture-v1/policy.zip \
  --video --output runs/linux-infer
```

首版不需要 CUDA。`--device` 暴露给已有兼容 PyTorch 环境的用户；锁定安装默认是 CPU，指定 `cuda` 不会自动安装 GPU 版本。

## 检查与排错

```bash
orca-hand doctor
orca-hand --help
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

- 模型找不到：先 bootstrap；从仓库运行，或设置 `ORCA_WORKSPACE`。
- 可选模块找不到：重新同步对应 extra。
- 输出目录已有轨迹：换一个 `--output`，保留之前的实验。
- 没有显示服务：使用 `--headless`；导出视频仍需要 OpenGL 后端。

原始视频、设备标定、训练日志和快照均被 Git 忽略。随仓库提供的示例模型与验证摘要位于 `artifacts/`。
