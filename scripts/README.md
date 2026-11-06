# 现有命令

在项目根目录运行。环境安装和同步步骤见 [使用说明](../使用说明.md)。

| 入口 | 用途 |
|---|---|
| `setup_env.ps1`、`run.ps1`、`check_env.py` | 重建、使用和检查锁定的 Windows 环境 |
| `download_earth_data.py`、`build_earth_reference.py`、`build_region_atlas.py` | 下载和生成 ETOPO／Natural Earth 对照 |
| `design_final_map.py`、`build_present_terrain.py` | 栅格化 v2、构建兼容地形与海岸候选 |
| `design_east_corridor.py`、`design_present_inland_sea.py` | 区域过程样板 |
| `build_globe.py` | 构建离线球面图志 |

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/build_present_terrain.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/design_east_corridor.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/build_globe.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python -m pytest
```

新引擎入口随首个区域样板实现。旧构造试验和批量阶段出图见 [研究归档](../tasks/tectonic_history.md)。
