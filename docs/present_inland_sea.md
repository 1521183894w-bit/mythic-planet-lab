# 主内海现世地理首轮

从 v2 的 2048×1024 格网重建主内海，采用中心 37.5°N、17.0°E 的局部等面积方位投影。区域覆盖 8,500 × 3,900 公里，像元 5 公里。全海域图、原图和候选对照、爱琴海同尺度对照见 [图集](../figures/design/v2/present_inland_sea/index.html)。

## 图层

- 海陆骨架为 v2 设计（`user_override`），区域海岸为有符号距离重采样（`interpolated`）。旧高程与沿用的水深均为 v2 示意（`visual_only`）。
- 山前外带和三处低地按已有的两岸造山、湾缘沉降、东湾前陆设定布置（`user_override`）。种子 602 提供原 v2 格网以下的初始扰动，受坡度和沉积低地调制（`procedural_constrained`）。
- 局部演算复用东方走廊的 Fastscape Fortran 2.8.4 和 RichDEM 2.4.3（`parameterized_physics`）。运行 16 个 25,000 模型年步长，沿单向汇流路径用河流功率律下切并做坡面扩散，模型参数沿用样板，时间和系数用于未标定的情景。岸线固定，沉积与海侵留待相应过程模型。
- 蓝虚线是格网上避陆的水面通道（`visual_only`），东端在东地峡西侧水域。水深、港口条件及潮汐待核算。地球图用已有 NOAA ETOPO 2022 60 角秒数据，重采样到相同像元（`interpolated`）。

方法和资料参考沿用 [东方走廊样板](east_corridor_sample.md)。

## 复现与结论

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python scripts\design_present_inland_sea.py
```

格网保存在项目内 `.cache/present_inland_sea.npz`。重建用时 34.51 秒，含 v2 的全球 A 层复算，另计缓存保存与制图时间。两次出图的三张 PNG 逐字一致。地球对照来自西部固定窗口。全球 B 层留待阶段验收，实际通航条件待核算。

相对原图，高程变化超过 150 米的区域为约 242 万平方公里，主要来自初始地势调整。演算相对初始地表的绝对改变量在陆地上的中位为 2.01 米、第 95 百分位为 54.67 米。地图可用于辨认既有地标关系。新地表仍缺少成组的岬湾、岛群、支脉和谷地，按停手规则归档候选并保留原 v2。 [独立评审](../reviews/present_inland_sea_2026-11-05_1.md)
