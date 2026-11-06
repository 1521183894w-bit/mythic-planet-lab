# 终图 v2 · 全球海岸复杂度候选

这轮覆盖 13 个沿海区域，分别处理主内海两岸、北方内海、北极群岛和陆表海、中部长海、东北活动陆缘、洲 C 西岸、洲 D 东岸、远方大陆东西岸的选定地段，以及极地大陆的两段岸线。其余岸段提供平直岸、开阔陆架和洋盆作对照。

## 方法与复现

`config/design/coasts_v2.yaml` 记录每个区域的成因假说、走向、宽度、深度或高度。主内海采用造山前缘、断块与弧后伸展假说。高纬岸段用既有古岩浆弧或地盾高地作为冰蚀出流的背景。洲 C 沉降陆表海、洲 D 夭折裂谷和远方东岸旧谷分别使用分支深湾、断块接力槽和低缓河口。河口、峡湾和断层入海的地貌分类参考 [NOAA 海岸河口地貌资料](https://oceanservice.noaa.gov/education/tutorial_estuaries/est04_geology.html) 和 [华盛顿州自然资源部冰蚀岸地质说明](https://dnr.wa.gov/puget-sound-and-coastal-geology)。

球面轴线定义槽谷及岩脊，有限幅度的侧壁起伏只作用在岸段走廊内部。海陆分类取走廊与旧岸的交切，计算上要求新挖水域接入原有外海或主内海一侧水体。沿岸高程采用与同一走廊相容的有限过渡。陆上淡水湖和咸湖、咸内海、T1/T2 海峡、神怒湾、涌潮湾、I4 岛核和浮冰架是保护区。新淹没旧陆地的格点不分配洋壳代理年龄。新增 `coast_design_region` 是 `user_override`，可由区号追溯走廊假说。交切后的地表为 `procedural_constrained`。谷系历史、真实航道水深与冰流均待后续过程或局部高分辨率数据核实。

在项目根目录运行。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/build_present_terrain.py --coasts
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/make_figures.py --world data/worlds/present_v2/terrain_coasts_2048x1024.nc --stage 1 --out figures/stage1/present_v2_coasts
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/present_terrain_evidence.py --world data/worlds/present_v2/terrain_coasts_2048x1024.nc --figures figures/stage1/present_v2_coasts
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/build_globe.py --terrain data/worlds/present_v2/terrain_coasts_2048x1024.nc --output figures/stage1/present_v2_coasts/globe/index.html
```

标准图是 `figures/stage1/present_v2_coasts/index.html`。对照画廊在同目录的 `evidence/index.html`，从现行 v2 高程、岸线候选和 ETOPO 同分辨率格网采样。需要先按 [现世地形底图](present_terrain.md) 重建原海陆与高程。候选单独存为 `terrain_coasts_2048x1024.nc`，原入口仍可复现 v2 底稿。

## 实测

| 核对项 | 2048×1024 结果 |
|---|---|
| A 层 | 7/7。海洋 71.6059%。主大陆约 1.0546 亿 km²，远方大陆约 2142.1 万 km²，极地大陆约 1279.2 万 km²。 |
| 主内海 | 主体 437.1 万 km²，A 层仍识别一处通外洋海峡。锁定的 T1/T2 格点和水深逐格相同。窄口判定将若干局部湾或岛间水体另列为内侧连接，统计从 1 处变为 6 处，代表形态学切分。I4 仍是独立岛。 |
| B1 现世阶段尺度 | 三块大陆的 100、200 km 半岛、海湾指数及分形维数均达到原门槛。主大陆 100 km 半岛指数为 1.6202%，门槛 1.6179%，余量很小。 |
| B2 | 主大陆距海中位数 611.8→520.8 km，超 1000 km 的陆地占比 30.29%→24.43%。三块大陆的两个距海指标达标。 |
| 50 km 诊断 | 主大陆及远方大陆的半岛、海湾指数仍低于参考，极地大陆海湾指数略低于参考。综合显示 7/9、7/9、8/9。 |
| 范围 | 共 20,953 格改变海陆类别，按球面面积计新增陆地 204.6 万 km²，淹没原陆地 251.3 万 km²。地形变化限于设计走廊或离新旧海陆变化 145 km 内。四条海沟轴均未穿陆。 |
| 稳定与性能 | 两次 2048×1024 重建的八层数据逐格一致。串行实测约 25–28 秒，NetCDF 约 13.68 MiB。现世相关测试 63 项通过。 |

50 km 结果只描述候选现在的限制。海岸线和岩性靠同尺度图及固定岸段检查验收。阶段一整体还须满足独立评审与用户确认要求。
