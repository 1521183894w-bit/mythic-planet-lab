"""原终图 v2 的现世主内海区域草图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
from scipy import ndimage as ndi
from shapely import Polygon, contains_xy
from skimage.graph import route_through_array
import yaml

from design_east_corridor import erode, STEP_YEARS
from design_east_corridor_figures import ETOPO, terrain_rgb
from design_final_map import LatLonGrid, apply_options, build, elevation, code_version
from mythic.figures.regional import laea_forward, laea_window, sample_global
from mythic.figures.style import setup_fonts
from mythic.metrics.compute import compute_all, load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/design/final_map_v2.yaml"
OUT = ROOT / "figures/design/v2/present_inland_sea"
CENTER = (37.5, 17.0)
PIXEL = 5
WIDTH, HEIGHT = 8500, 3900
EXTENT = (-WIDTH / 2, WIDTH / 2, -HEIGHT / 2, HEIGHT / 2)
STEPS = 16
SEED = 602
INK = "#183340"
PAPER = "#faf8f2"

# 区域级沉降轴。下游谷地和更小尺度的沟壑交给侵蚀。
PLAINS = [
    # 内海西北湾后缘的弧后低地
    ([(-3.2, 43.4), (0.0, 43.0), (3.4, 42.4)], 115, 100),
    # 内海南湾湾头的陆缘沉降低地
    ([(12.0, 23.1), (13.7, 24.8), (16.4, 26.6)], 200, 90),
    # 东湾前陆的斜向低地
    ([(47.6, 30.8), (50.3, 30.6), (53.1, 30.0), (56.9, 27.9)], 130, 110),
]
# 山前外带，走向沿原有造山带。数值依次为有效半宽和附加起伏（km、m）。
FRONTS = [
    ([(-24, 39.8), (-20, 40.8), (-17.8, 40.4), (-15.3, 38.8)], 59, 460),
    ([(-11, 41.5), (-6, 43.3), (-2, 45.0), (3.7, 46.6)], 67, 670),
    ([(2.5, 43.4), (6.4, 44.1), (10.2, 43.9)], 47, 540),
    ([(-19.0, 32.4), (-14.5, 31.0), (-9.1, 30.4), (-6.7, 31.2)], 62, 480),
    ([(-4.6, 30.4), (1.8, 30.0), (6.0, 29.3), (10.8, 29.1)], 63, 630),
    ([(24.0, 29.4), (29.5, 30.3), (34.5, 31.0), (38.1, 31.8)], 72, 700),
    ([(41.2, 31.2), (44.0, 31.7), (47.8, 33.0)], 55, 560),
    ([(49.8, 36.0), (53.4, 34.2), (58.6, 31.6), (62.0, 29.5)], 63, 720),
]

LABELS = [
    ("西大洋", -24.3, 34.0, "water"),
    ("西海峡", -15.2, 36.25, "callout", -17.0, 38.5),
    ("西北湾", 3.0, 42.2, "callout", 0.9, 45.0),
    ("西部群岛", -2.4, 37.45, "callout", -5.3, 39.4),
    ("微大陆岛", 5.3, 39.3, "callout", 7.9, 41.9),
    ("北岸山系", 8.1, 46.6, "land"),
    ("神怒湾", 17.0, 43.4, "callout", 18.5, 46.4),
    ("多湾半岛", 26.0, 39.1, "callout", 27.8, 42.9),
    ("北方内海", 38.2, 45.25, "water"),
    ("内海主盆", 32.8, 35.6, "water"),
    ("南岸山系", -6.0, 32.2, "land"),
    ("内海南湾", 16.6, 30.25, "water"),
    ("湾头低地", 15.6, 24.5, "land"),
    ("东地峡", 49.7, 33.0, "callout", 52.5, 35.7),
    ("东湾", 57.3, 27.55, "water"),
]


def project(points, center=CENTER):
    lon, lat = np.asarray(points).T
    return np.column_stack(laea_forward(*center, lat, lon))


def grid_window(center, width, height):
    lat, lon, _ = laea_window(*center, width, PIXEL)
    trim = (width - height) // (2 * PIXEL)
    return lat[trim:-trim], lon[trim:-trim]


def strip(xx, yy, points, width):
    """计算沿折线的连续横向权重。"""
    xy = project(points)
    distance2 = np.full((yy.size, xx.size), np.inf)
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        distance2 = np.minimum(distance2, (xx - x0 - t * dx)**2 + (yy - y0 - t * dy)**2)
    return np.exp(-.5 * distance2 / width**2)


def make_surface():
    design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf-8")), set())
    source = LatLonGrid(2048, 1024)
    world = build(design, source)
    src_land = world["land"]
    src_z = elevation(design, source, world)
    source_metrics = compute_all(src_z, src_land, cfg=load_config())
    lat, lon = grid_window(CENTER, WIDTH, HEIGHT)
    x = EXTENT[0] + (np.arange(lon.shape[1]) + .5) * PIXEL
    y = EXTENT[3] - (np.arange(lat.shape[0]) + .5) * PIXEL
    signed = ndi.distance_transform_edt(src_land) - ndi.distance_transform_edt(~src_land)
    land = sample_global(signed, lat, lon) > 0
    lake = (sample_global(world["lake"].astype(float), lat, lon) >= .5) & land
    coast = ndi.distance_transform_edt(land) * PIXEL
    old_z = sample_global(src_z, lat, lon)
    old_z = np.where(land, np.maximum(old_z, 1), np.minimum(old_z, -1))

    xx, yy = x[None, :], y[:, None]
    draft = old_z.copy()
    plains = np.zeros_like(draft)
    for line, width, base in PLAINS:
        weight = strip(xx, yy, line, width)
        plains = np.maximum(plains, weight)
        # 沉积低地在近岸最低，向内陆抬升。保留已存在的主要山脊。
        target = base + .19 * coast
        draft -= weight * np.clip((1800 - old_z) / 1400, 0, 1) * np.maximum(draft - target, 0)
    for line, width, relief in FRONTS:
        draft += strip(xx, yy, line, width) * relief * (1 - .8 * plains)

    rock = np.where(plains > .48, 2, 1).astype(np.uint8)
    for region in design["lithology"]:
        if region["id"] in {"LT6", "LT9"}:
            rock[contains_xy(Polygon(region["poly"]), lon, lat)] = 5
    rock[~land] = 0
    # 种子只补低于源格网的初始起伏，幅度随坡度及软硬岩变化。
    rng = np.random.default_rng(SEED)
    perturbation = ndi.gaussian_filter(rng.standard_normal(draft.shape), .65)
    perturbation /= perturbation.std()
    dy, dx = np.gradient(draft, PIXEL)
    draft += perturbation * (4 + np.clip(np.hypot(dx, dy), 0, 26)) * (1 - .7 * plains)
    ramp = 12 + 46 * plains
    draft = np.where(land, np.maximum(draft, 4) * np.tanh(coast / ramp), old_z)
    no_red = np.zeros_like(land)
    new_z = erode(draft, land, rock, draft, no_red, PIXEL, STEPS)
    # 咸水湖保留原设计的水域显示。
    new_z[lake] = old_z[lake]

    # 纯几何海上走廊，避开陆地及内流湖；远离岸边的像元代价更低。
    water_dist = ndi.distance_transform_edt(~land) * PIXEL
    cost = np.where(land, np.inf, 1 + 40 / np.maximum(water_dist, PIXEL))
    checkpoints = [(-25.0, 35.8), (-14.7, 36.15), (11.5, 35.3), (33.8, 37.6), (47.5, 34.7)]
    cells = []
    for px, py in project(checkpoints):
        row, col = abs(y - py).argmin(), abs(x - px).argmin()
        if land[row, col]:
            iy, ix = np.where(~land[max(0, row-9):row+10, max(0, col-9):col+10])
            iy, ix = iy + max(0, row-9), ix + max(0, col-9)
            nearest = np.argmin((iy - row)**2 + (ix - col)**2)
            row, col = int(iy[nearest]), int(ix[nearest])
        cells.append((row, col))
    segments = []
    for start, end in zip(cells[:-1], cells[1:]):
        path, route_cost = route_through_array(cost, start, end, fully_connected=True)
        assert np.isfinite(route_cost), (start, end)
        segments.extend(path[:-1])
    segments.append(cells[-1])
    rows, cols = np.asarray(segments).T
    assert not land[rows, cols].any()
    change = abs(new_z - old_z)
    keep = land & ~lake
    assert np.array_equal(new_z > 0, land)
    assert np.array_equal(new_z[~land], old_z[~land])
    metrics = {
        "input": DESIGN.relative_to(ROOT).as_posix(),
        "code_version": code_version(),
        "method": "v2_present_inland_sea_v1",
        "source_grid": "2048×1024",
        "resolution_km": PIXEL,
        "shape": list(land.shape),
        "center_lat_lon": CENTER,
        "extent_km": EXTENT,
        "layers": {
            "source_coast": {"method_class": "user_override", "units": "mask"},
            "regional_coast": {"method_class": "interpolated", "units": "mask"},
            "old_elevation_and_depth": {"method_class": "visual_only", "units": "m"},
            "initial_lowlands_foothills_and_rock": {"method_class": "user_override", "units": "m / class"},
            "initial_subgrid_perturbation": {"method_class": "procedural_constrained", "seed": SEED, "units": "m"},
            "incision_and_diffusion": {"method_class": "parameterized_physics", "version": "Fastscape Fortran 2.8.4 + RichDEM 2.4.3", "units": "m"},
            "geometric_passage": {"method_class": "visual_only", "units": "km"},
            "earth_reference": {"method_class": "interpolated", "version": "NOAA ETOPO 2022 v1 60s", "units": "m"},
        },
        "steps": STEPS,
        "step_model_years": STEP_YEARS,
        "global_source_ocean_pct": float((1 - (source.cell_area_km2 * src_land).sum() / source.total_area_km2) * 100),
        "regional_mask_difference_cells": int(np.count_nonzero((new_z > 0) != (old_z > 0))),
        "regional_land_change_over_150m_km2": int(np.count_nonzero(keep & (change > 150))) * PIXEL**2,
        "regional_new_max_land_m": round(float(new_z[keep].max()), 2),
        "land_cells_on_geometric_passage": int(land[rows, cols].sum()),
        "source_A": {k: bool(v["pass"]) for k, v in source_metrics["A"].items()},
        "source_A_note": "原 v2 全球设计格网的重算结果。区域试验未重跑全球 B 层。",
        "erosion_absolute_change_land_m_p50_p95": np.round(
            np.percentile(abs(new_z - draft)[keep], [50, 95]), 2
        ).tolist(),
        "notes": "陆缘几何和低地/褶皱外带为现世设计；侵蚀时长与系数为未标定的演示情景。海深为旧终图示意，海路只核对格网水面连通。",
    }
    return metrics, {
        "lat": lat, "lon": lon, "land": land, "lake": lake, "rock": rock,
        "old_z": old_z, "draft": draft, "new_z": new_z, "x": x, "y": y,
        "route_x": x[cols], "route_y": y[rows],
    }


def read_earth(center, width, height):
    from netCDF4 import Dataset

    lat, lon = grid_window(center, width, height)
    with Dataset(ETOPO) as ds:
        slat, slon = (np.asarray(ds[name][:]) for name in ("lat", "lon"))
        ii = np.flatnonzero((slat > lat.min() - .06) & (slat < lat.max() + .06))
        jj = np.flatnonzero((slon > lon.min() - .06) & (slon < lon.max() + .06))
        patch = np.asarray(ds["z"][ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=float)
    return ndi.map_coordinates(patch, [(lat - slat[ii[0]]) / (slat[1] - slat[0]),
                                       (lon - slon[jj[0]]) / (slon[1] - slon[0])], order=1)


def paint(ax, rgb, extent, window, km, title, coords):
    ax.imshow(rgb, extent=extent, interpolation="nearest", zorder=0)
    ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[], aspect="equal")
    for spine in ax.spines.values():
        spine.set_color("#a7babd")
    x0, x1, y0, y1 = window
    xr, yr = x1 - x0, y1 - y0
    px, py = x0 + xr * .038, y0 + yr * .063
    ax.plot([px, px + km], [py, py], color=INK, linewidth=2, zorder=6)
    ax.text(px, py + yr*.026, f"{km} km", fontsize=8, color=INK, zorder=6,
            bbox={"fc": PAPER, "ec": "none", "alpha": .82, "pad": 1})
    ax.set_title(title, loc="left", fontsize=12, color=INK, weight="bold", pad=7)
    lat, lon, x, y = coords
    cs = ax.contour(x, y, lon, levels=np.arange(-40, 71, 10), colors="#567d87", linewidths=.38, alpha=.45)
    ts = ax.contour(x, y, lat, levels=np.arange(20, 61, 5), colors="#567d87", linewidths=.38, alpha=.45)
    ax.clabel(cs, fontsize=7, fmt=lambda v: f"{abs(v):g}°{'W' if v < 0 else 'E' if v > 0 else ''}")
    ax.clabel(ts, fontsize=7, fmt=lambda v: f"{v:g}°N")


def header(fig, title, sub):
    fig.text(.032, .962, title, fontsize=22, weight="bold", color=INK, va="top")
    fig.text(.033, .905, sub, fontsize=10.5, color="#435c65", va="top")


def render(data, meta, out):
    from matplotlib import patheffects as pe

    setup_fonts()
    coords = tuple(data[name] for name in ("lat", "lon", "x", "y"))
    old_z, new_z, land, lake = (data[name] for name in ("old_z", "new_z", "land", "lake"))
    colors = [terrain_rgb(z, np.where(lake, -70, old_z), land & ~lake, PIXEL) for z in (old_z, new_z)]
    halo = [pe.withStroke(linewidth=2.8, foreground=PAPER)]
    foot = f"v2 · 5 km 区域格网 · 原地形与水深 visual_only · 区域海岸 interpolated · 初始地形 user_override · Fastscape 地表 parameterized_physics · {meta['code_version']}"

    fig = plt.figure(figsize=(13.9, 13.7), facecolor=PAPER)
    header(fig, "主内海｜现世地理初稿", "相同海岸底图、投影、格距、色带和阴影。先对照西口至东地峡的整体地势。")
    for j, (rgb, title) in enumerate(zip(colors, ["原终图 v2  ·  区域重采样", "现世草图  ·  两岸山前与低地"])):
        ax = fig.add_axes([.034, .467 - j * .403, .932, .378])
        paint(ax, rgb, EXTENT, EXTENT, 500, title, coords)
    fig.text(.035, .037, foot, fontsize=8.7, color=INK)
    fig.savefig(out / "01_before_after.png", dpi=155)
    plt.close(fig)

    fig = plt.figure(figsize=(15.4, 9.0), facecolor=PAPER)
    header(fig, "主内海｜从西海峡到东地峡", "现世创作底图草案　·　山地、低地、岛群与东西海上通道")
    ax = fig.add_axes([.032, .123, .936, .727])
    paint(ax, colors[1], EXTENT, EXTENT, 500, "原终图 v2 海陆骨架  /  现世地表草图", coords)
    ax.plot(data["route_x"], data["route_y"], color="#1b6280", lw=1.65, linestyle=(0, (5, 3)), zorder=4,
            path_effects=[pe.withStroke(linewidth=3.3, foreground="#d9f0ed")])
    for name, lon, lat, kind, *text_at in LABELS:
        point = project([(lon, lat)])[0]
        color = "#18465c" if kind in {"water", "callout"} else "#573e2e"
        if kind == "callout":
            ax.annotate(name, point, xytext=project([text_at])[0], ha="center", va="center",
                        fontsize=8.8, color=color, weight="bold", path_effects=halo, zorder=7,
                        arrowprops={"arrowstyle": "-", "color": color, "linewidth": .7})
        else:
            ax.text(*point, name, ha="center", va="center", fontsize=10, weight="bold",
                    color=color, path_effects=halo, zorder=7)
    ax.legend(handles=[Patch(color="#a6ad7e", label="低地"), Patch(color="#a99076", label="中低山"),
                       Patch(color="#e1d2bd", label="高山"), Patch(color="#95c1d1", label="水域")],
              loc="lower right", fontsize=8.5, facecolor=PAPER, framealpha=.94, ncol=4)
    fig.text(.033, .094, "蓝色虚线为格网上的海上走廊（visual_only）。以陆地避让选线，水深和港口条件待核。", fontsize=9.4, color=INK)
    fig.text(.033, .063, foot, fontsize=8.7, color=INK)
    fig.savefig(out / "02_present_map.png", dpi=165)
    plt.close(fig)

    width, height = 2100, 1300
    earth_center = (38.1, 23.6)
    earth = read_earth(earth_center, width, height)
    ea_lat, ea_lon = grid_window(earth_center, width, height)
    ex = -width / 2 + (np.arange(earth.shape[1]) + .5) * PIXEL
    ey = height / 2 - (np.arange(earth.shape[0]) + .5) * PIXEL
    cx, cy = project([(8.8, 39.1)])[0]
    earth_extent = (-width/2, width/2, -height/2, height/2)
    detail = (cx-width/2, cx+width/2, cy-height/2, cy+height/2)
    fig = plt.figure(figsize=(15.6, 6.9), facecolor=PAPER)
    header(fig, "同尺度地球对照｜内海西部与爱琴海", "每窗 2,100 × 1,300 km · 5 km 像元 · 相同投影类型、高程设色和阴影")
    earth_rgb = terrain_rgb(earth, earth, earth > 0, PIXEL)
    for j, (rgb, extent, window, title, grid) in enumerate([
        (colors[1], EXTENT, detail, "v2  ·  西北湾、微大陆岛与长半岛", coords),
        (earth_rgb, earth_extent, earth_extent, "地球  ·  爱琴海与周边陆缘", (ea_lat, ea_lon, ex, ey)),
    ]):
        ax = fig.add_axes([.035 + .473 * j, .157, .453, .654])
        paint(ax, rgb, extent, window, 250, title, grid)
    fig.text(.034, .103, "地球参考 NOAA ETOPO 2022（interpolated）。草图两岸是地形方案，浅海与盆底使用终图示意值。", fontsize=9, color=INK)
    fig.savefig(out / "03_earth_comparison.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/present_inland_sea.npz")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metrics.json"
    if args.render_existing:
        metrics = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(args.data) as arrays:
            render(arrays, metrics, args.out)
        return
    start = time.perf_counter()
    metrics, arrays = make_surface()
    metrics["model_seconds"] = round(time.perf_counter() - start, 2)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.data, **arrays)
    meta_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    render(arrays, metrics, args.out)


if __name__ == "__main__":
    main()
