# ORCA v1 掌内定向：实验策略

训练 601584 个环境步。任务是将自由笔长轴转向 ±90° 并保持 0.5 秒，允许掌面支撑。

独立评估 32 回合：成功 1 次、掉落 6 次。此 checkpoint 用于复现试验，尚未达到可靠执行任务的水平，也未用于真机。

复现：

```bash
.venv/bin/python scripts/pen_task.py evaluate --checkpoint artifacts/pretrained/pen-v1-experimental/policy.zip --video --output runs/pen-reproduction
```

完整配置见相邻 config.json；训练过程与定义见 docs/software/pen.md。
