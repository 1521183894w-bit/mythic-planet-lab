# 工具小试

保留与新主线直接相关的存储、侵蚀和汇流验证。原始报告与实测见 [工具评估](../docs/tool_evaluation.md)。旧球面网格、板块运动和整套阶段试验可从 [清理前的提交](../tasks/tectonic_history.md) 获取。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python -u spikes/stage0_tools/03_storage.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python -u spikes/stage0_tools/05_landscape_evolution.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python -u spikes/stage0_tools/06_hydrology.py
```

汇流试验使用侵蚀试验的结果。派生数据写入 D 盘项目内的忽略目录。
