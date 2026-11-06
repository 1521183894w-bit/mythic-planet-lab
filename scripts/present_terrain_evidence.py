"""绘制现世地形候选的旧版/地球对照与固定岸段的构造假说。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import yaml

from build_present_terrain import COASTS, CONFIG, DESIGN, OUTPUT, ROOT, trench_path
from design_final_map import apply_options, build, densify, elevation, to_lonlat
from mythic.figures.netcdf import open_netcdf
from mythic.figures.region_panels import _window
from mythic.figures.regional import draw_elevation_panel, laea_forward, sample_global
from mythic.figures.style import elevation_cmap_norm, setup_fonts, stamp
from mythic.figures.world import World, load_earth, load_world_nc

DEFAULT_FIGURES = ROOT / "figures/stage1/present_v2_ocean"
REQUIRED = ("r2", "r6", "r7", "r8")
STYLES = {
    "mountains": ("#7e3928", "山系／裂谷肩"),
    "gulfs": ("#278782", "海湾设计轴"),
    "islands": ("#934e9e", "岛弧／岛链"),
    "trenches": ("#1e4489", "海沟假说"),
    "ridges": ("#e98222", "扩张轴假说"),
    "marine_basins": ("#174a5f", "海盆控制线"),
    "interior_provinces": ("#807024", "内陆丘陵带"),
}
LITH_COLORS = ["#e0dace", "#e6c576", "#f18d5b", "#b57466", "#c29bc3", "#89a692", "#9aaab7", "#ac9263"]


def traces(design, terrain):
    for category in STYLES:
        for spec in design.get(category, terrain.get(category, [])):
            if category == "islands" and spec["type"] not in {"arc", "hotspot"}:
                continue
            if "segments" in spec:
                yield from ((category, spec["id"], segment) for segment in spec["segments"])
            elif "line" in spec:
                path = trench_path(spec["line"]) if category == "trenches" else spec["line"]
                yield category, spec["id"], path


def draw_traces(ax, sources, project, extent, styles, label=False):
    for category, key, path in sources:
        lon, lat = to_lonlat(densify(path, 18)[0])
        x, y = project(lon, lat)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x0, x1, y0, y1 = extent
        keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not keep.any():
            continue
        # 切开图幅之外的段与跨接缝的段。
        x[(~keep) | (np.abs(np.diff(x, prepend=x[0])) > 0.35 * (x1 - x0))] = np.nan
        ax.plot(x, y, lw=1.15, color=styles[category][0], alpha=0.94)
        if label:
            visible = np.flatnonzero(np.isfinite(x))
            if visible.size > 4:
                i = visible[len(visible) // 2]
                ax.annotate(key, (x[i], y[i]), xytext=(2, 3), textcoords="offset points",
                            color=styles[category][0], size=6, clip_on=True)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])


def legends(fig, design, styles):
    from matplotlib.patches import Patch
    labels = [Line2D([0], [0], color=color, lw=1.6, label=title) for color, title in styles.values()]
    kinds = list(dict.fromkeys(s["kind"] for s in design["lithology"]))
    names = {"carbonate": "碳酸盐岩", "sandstone": "砂岩", "redbed": "红层", "evaporite": "蒸发岩",
             "craton": "古地盾", "granite": "花岗岩", "ophiolite": "蛇绿岩"}
    labels.extend(Patch(facecolor=c, label=n) for c, n in
                  zip(LITH_COLORS, ["未指定岩性"] + [names[k] for k in kinds]))
    fig.legend(handles=labels, ncols=7, loc="lower center", bbox_to_anchor=(0.5, .065), fontsize=8)
    return np.array([0] + [kinds.index(s["kind"]) + 1 for s in design["lithology"]], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--coasts", type=Path, default=COASTS, help="用于核对现世岸段设计来源")
    args = parser.parse_args()
    world = load_world_nc(args.world)
    earth = load_earth(world.resolution)
    design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf-8")), set())
    terrain = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    regions = json.loads((args.figures / "regions.json").read_text(encoding="utf-8"))["regions"]
    out = args.figures / "evidence"
    out.mkdir(exist_ok=True)
    setup_fonts()
    with open_netcdf(args.world) as ds:
        for attr, source in (("design_sha256", DESIGN), ("terrain_config_sha256", CONFIG)):
            if ds.attrs[attr] != sha256(source.read_bytes()).hexdigest():
                raise ValueError("高程文件与当前设计或地形配置不一致，请重建高程")
        lith = ds.lithology_province.values
        coast_hash = ds.attrs.get("coast_config_sha256")
    styles = STYLES.copy()
    sources = list(traces(design, terrain))
    files = []
    if coast_hash:
        raw = args.coasts.read_bytes()
        if coast_hash != sha256(raw).hexdigest():
            raise ValueError("岸段设计配置已变更，请重建高程")
        for region in yaml.safe_load(raw)["regions"]:
            for key in ("cuts", "headlands"):
                sources.extend((key, region["id"], item["line"]) for item in region.get(key, ()))
        styles.update({"cuts": ("#286780", "溺谷／冰蚀槽假说"), "headlands": ("#6a8227", "岩脊／岬角假说")})
        old = load_world_nc(OUTPUT)
        old.display_name = "现行 v2 岸线"
        if old.resolution != world.resolution:
            raise ValueError("原始底图和岸线候选格距应一致")
    else:
        geo = build(design, world.grid)
        old = World("design_v2", "原终图 v2", elevation(design, world.grid, geo),
                    geo["land"], "visual_only", "final_map_v2.yaml", radius_km=world.radius_km)

    comparisons = [
        ("01_inland_and_margin.png", "主内海西段和远方大陆西缘", [
            ("主内海西段 / 爱琴海", (39, 0), (38, 24)),
            ("远方大陆西缘 / 秘鲁—智利陆缘", (-12, -122), (-15, -78))]),
        ("02_interior_and_plateau.png", "内陆低地和高原边缘", [
            ("洲 C 内陆 / 刚果盆地", (12, -9), (0, 21)),
            ("东方大高原与东缘 / 青藏高原东缘", (33, 96), (32, 99))]),
    ]
    if coast_hash:
        comparisons = [
            ("01_inland_sea.png", "主内海两岸", [("西北湾与微大陆岛", (40, 2), (38.5, 24.5)),
                                                     ("内海南岸褶皱海岸", (33.2, 28.5), (38.6, 13.8))]),
            ("02_northern_seas.png", "北方冰缘与陆表海", [("西北冰川半岛", (65, -17), (63, 10)),
                                                          ("北方陆表海", (69, 73), (62, 22))]),
            ("03_south_continents.png", "洲 C 与洲 D 的海湾", [("洲 C 沉降陆表海", (-3, -18), (-2, -51)),
                                                               ("洲 D 夭折裂谷", (-4, 45), (25, -110))]),
            ("04_far_continent.png", "远方大陆的两种岸形", [("东岸溺谷", (-7, -80), (38, -75.6)),
                                                              ("南端冰蚀岸", (-45, -103), (-47, -74))]),
            ("05_polar_coasts.png", "极地长半岛与岛弧", [("极地长半岛", (-68, -69), (-65, -63)),
                                                          ("极地东半岛", (-65, 108), (-66, 115))]),
            ("06_north_east.png", "中部长海与活动陆缘", [("古缝合带入海", (63, 10), (62, 22)),
                                                           ("东北沿岸与弧后海", (46, 139), (43, 141))]),
        ]
    for file, title, rows in comparisons:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.subplots_adjust(left=.025, right=.94, bottom=.115, top=.91, wspace=.035, hspace=.15)
        for axes, (caption, center, earth_center) in zip(axs, rows):
            for ax, field, c in zip(axes, (old, world, earth), (center, center, earth_center)):
                e, land, lat, lon, ext, px = _window(field, c, 2200)
                draw_elevation_panel(ax, e, land, lat, lon, ext, px, field.display_name, c)
            axes[0].set_ylabel(caption)
        cmap, norm = elevation_cmap_norm()
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes((.943, .21, .011, .60)),
                     label="高程 / 水深（m）")
        fig.suptitle(f"{title} · 2200 km 窗口 · 局部等面积方位投影")
        stamp(fig, world, f"旧地形 {old.method_class} / 候选 procedural_constrained / 地球 interpolated（ETOPO 2022）")
        fig.savefig(out / file, dpi=124)
        plt.close(fig)
        files.append((file, title))

    fig, ax = plt.subplots(figsize=(15, 8.2))
    fig.subplots_adjust(left=.055, right=.97, bottom=.19, top=.90)
    codes = legends(fig, design, styles)
    extent = (-180, 180, -90, 90)
    ax.imshow(np.ma.masked_where(~world.land, codes[lith]), extent=extent, cmap=ListedColormap(LITH_COLORS),
              vmin=0, vmax=len(LITH_COLORS)-1, interpolation="nearest")
    ax.set_facecolor("#d6e6ee")
    draw_traces(ax, sources, lambda lo, la: (lo, la), extent, styles)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))
    ax.grid(alpha=.28)
    ax.set(xlabel="经度（°）", ylabel="纬度（°）")
    fig.suptitle("全球构造走向与已有岩性设计假说")
    stamp(fig, world, "user_override", ["等距圆柱投影，赤道 30° 经差约 3336 km。浅灰为岩性待定的陆地。"])
    file = "07_global_hypotheses.png" if coast_hash else "03_global_hypotheses.png"
    fig.savefig(out / file, dpi=124)
    plt.close(fig)
    files.append((file, "全球岩性与构造假说"))

    for record in regions:
        key = record["key"]
        if key[:2] not in REQUIRED or record["missing"]:
            continue
        center = record["center_lat_lon"]
        e, land, lat, lon, ext, px = _window(world, center, record["size_km"])
        fig, axes = plt.subplots(1, 2, figsize=(13.7, 7.5))
        fig.subplots_adjust(left=.025, right=.975, bottom=.19, top=.9, wspace=.035)
        draw_elevation_panel(axes[0], e, land, lat, lon, ext, px, "现世高程（m）", center)
        draw_elevation_panel(axes[1], e, land, lat, lon, ext, px, "海陆与同窗口构造假说", center)
        codes = legends(fig, design, styles)
        geol = sample_global(codes[lith], lat, lon, order=0)
        axes[1].imshow(np.ma.masked_where(land < .5, geol), extent=ext, cmap=ListedColormap(LITH_COLORS),
                       vmin=0, vmax=len(LITH_COLORS)-1, interpolation="nearest", alpha=.66)
        draw_traces(axes[1], sources, lambda lo, la: laea_forward(*center, la, lo), ext, styles, label=True)
        fig.suptitle(f"{key[:2]} · {record['title']} · 标准图固定窗口")
        stamp(fig, world, "高程 procedural_constrained / 设计假说 user_override",
              ["轴线、岩性与岸形是否相符由评审判断。"])
        file = f"{key}_evidence.png"
        fig.savefig(out / file, dpi=124)
        plt.close(fig)
        files.append((file, f"{key[:2]} {record['title']}"))

    body = ["<!doctype html><html lang=zh-CN><meta charset=utf-8><title>现世地形 · 对照与构造假说</title>",
            "<style>body{font:16px system-ui;max-width:1280px;margin:auto;padding:20px}img{width:100%}</style>",
            "<h1>终图 v2 · 对照与构造假说</h1><p><a href='../index.html'>标准图组</a> · "
            "<a href='../globe/index.html'>离线球面图志</a></p>"]
    body.extend(f"<h2>{title}</h2><a href='{file}'><img src='{file}' alt='{title}' loading='lazy'></a>"
                for file, title in files)
    (out / "index.html").write_text("\n".join(body), encoding="utf-8")
    print(f"{len(files)} 张对照与证据图 · {out}")


if __name__ == "__main__":
    main()
