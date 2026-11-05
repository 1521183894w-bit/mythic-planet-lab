"""终图 v2 东方走廊区域样板。区域坡降为设计，河谷由侵蚀与汇流生成。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import fastscapelib_fortran as fs
from fastscape.processes.flow import _flow_accumulate_sd
from numba import njit
import numpy as np
import richdem as rd
from scipy import ndimage as ndi
from shapely import Polygon, contains_xy
import yaml

from design_final_map import LatLonGrid, apply_options, build, elevation
from design_civilization_zones import EAST_ROUTE
from mythic.figures.regional import laea_forward, laea_window, sample_global

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/design/final_map_v2.yaml"
OUT = ROOT / "figures/design/v2/east_corridor"
CENTER = (33.2, 111.0)
WIDTH, HEIGHT = 3200, 2440
SEED = 296
STEP_YEARS = 25000
HIGHSTAND_M = 24
MAIN_KM2 = 4000
STREAM_KM2 = 96
ROCKS = [
    ("海域", "#cce3ea", 0),
    ("山地基底与变质岩", "#a99aad", 2.5e-6),
    ("前陆砂岩与泥岩", "#d8c58e", 5e-6),
    ("红层泥岩与粉砂岩", "#dc8d71", 6.5e-6),
    ("红层砂岩与砾岩", "#b35341", 3.0e-6),
    ("灰岩与白云岩", "#83a991", 3.3e-6),
    ("花岗岩", "#c397bb", 2.7e-6),
    ("谷地与沿岸冲积物", "#f1e4b9", 5e-6),
]


def source_surface(design, pixel):
    source = LatLonGrid(2048, 1024)
    geography = build(design, source)
    land0 = geography["land"]
    z0 = elevation(design, source, geography)
    lat, lon, bounds = laea_window(*CENTER, WIDTH, pixel)
    trim = round((WIDTH - HEIGHT) / (2 * pixel))
    lat, lon = lat[trim:-trim], lon[trim:-trim]
    extent = (bounds[0], bounds[1], -lat.shape[0] * pixel / 2, lat.shape[0] * pixel / 2)
    x = extent[0] + (np.arange(lat.shape[1]) + .5) * pixel
    y = extent[3] - (np.arange(lat.shape[0]) + .5) * pixel
    signed = ndi.distance_transform_edt(land0) - ndi.distance_transform_edt(~land0)
    # 取同一终图海岸，平滑原格网的阶梯。海面在有符号距离的零线上。
    land = ndi.gaussian_filter(sample_global(signed, lat, lon), 6 / pixel) > 0
    coast_km = ndi.distance_transform_edt(land) * pixel
    offshore_km = ndi.distance_transform_edt(~land) * pixel
    z = ndi.gaussian_filter(sample_global(z0, lat, lon), 7 / pixel)

    # 一个几百公里宽的沉降低地，由已标示的东方走廊控制位置。
    points = [(100.6, 31.2), *EAST_ROUTE]
    cx, cy = laea_forward(*CENTER, [p[1] for p in points], [p[0] for p in points])
    cross = y[:, None] - np.interp(x, cx, cy)
    lower = np.interp(x, [cx[0], cx[1], cx[-1]], [410, 305, 6]) + .0029 * cross**2
    weight = np.clip((x - cx[0] + 115) / 230, 0, 1) * np.exp(-.5 * (cross / 240)**2)
    weight *= np.clip((x - cx[-1] - 55) / -110, 0, 1)
    weight *= np.clip((1300 - z) / 800, 0, 1)
    z -= weight * np.maximum(z - lower, 0)

    # 东麓褶皱带及沿岸古山的残丘，给近乎平坦的终图底面补上区域分水岭。
    for lo, la, az, along, across, height in [
        (109.8, 35.7, 30, 255, 66, 195),
        (114.3, 37.8, 22, 250, 85, 265),
        (116.6, 35.8, 41, 132, 48, 138),
        (109.4, 28.4, 18, 210, 65, 145),
        (117.9, 31.3, 54, 122, 46, 175),
    ]:
        ax, ay = laea_forward(*CENTER, la, lo)
        xx, yy = x[None, :] - ax, y[:, None] - ay
        a = np.deg2rad(az)
        u = (xx * np.cos(a) + yy * np.sin(a)) / along
        v = (-xx * np.sin(a) + yy * np.cos(a)) / across
        z += height * np.exp(-.5 * (u * u + v * v)) * np.clip((1500 - z) / 1000, 0, 1)
    # 延续东南沿岸古山的走向，在平原出口两侧留下有高差的丘陵，
    # 给主河口保留低地。河谷与海岸具体形状由下切和海侵决定。
    coastal_hills = np.zeros_like(z)
    for lo, la, az, along, across, height in [
        (119.3, 32.2, 59, 133, 43, 380),
        (120.0, 36.9, 43, 165, 46, 335),
        (118.4, 35.7, 31, 120, 28, 190),
    ]:
        ax, ay = laea_forward(*CENTER, la, lo)
        xx, yy = x[None, :] - ax, y[:, None] - ay
        a = np.deg2rad(az)
        u = (xx * np.cos(a) + yy * np.sin(a)) / along
        v = (-xx * np.sin(a) + yy * np.cos(a)) / across
        coastal_hills += height * np.exp(-.5 * (u * u + v * v))
    coastal_hills *= 1 - np.exp(-.5 * (cross / 55)**2)
    z += coastal_hills
    # 岩省分界是区域尺度草案，独立于后来侵蚀的地表起伏。
    basement_edge = 101.5 + .95 * np.cos((lat - 29.6) * .54) + .45 * np.sin((lat - 31.7) * 1.05)
    rock = np.where(lon < basement_edge, 1, 2).astype("uint8")
    units = {f["id"]: contains_xy(Polygon(f["poly"]), lon, lat) for f in design["lithology"]
             if f["id"] in {"LT1", "LT4", "LT10"}}
    # 缓倾的红层接触面。河谷切过接触面可露出下面的软层。
    contact = 305 + 23 * (lon - 105.8) + 22 * (lat - 31.1)
    rock[units["LT1"]] = 5
    rock[units["LT10"]] = 6
    rock[units["LT4"]] = np.where(z[units["LT4"]] >= contact[units["LT4"]], 4, 3)
    rock[~land] = 0
    # 只补终图源格网未解析的起伏。坚硬山岩保留较强的微起伏，
    # 在盆地软岩中衰减，海岸处衰减到零；河网由后续下切产生。
    rng = np.random.default_rng(SEED)
    texture = ndi.gaussian_filter(rng.standard_normal(z.shape), (2.0 / pixel, 4.4 / pixel))
    texture /= texture.std()
    amplitudes = np.array([0, 24, 7, 5, 12, 17, 21, 0])
    dy, dx = np.gradient(z, pixel)
    # 陡坡和沿海硬岩丘陵中的初始微地形更明显，平原和海岸尖端衰减。
    scarp = np.clip(np.hypot(dx, dy) * 1.4, 0, 48)
    hills = np.clip(coastal_hills * .22, 0, 60)
    z += texture * (amplitudes[rock] + scarp + hills) * np.tanh(coast_km / (28 - hills / 4))
    soft_shelf = np.exp(-.5 * ((x[None, :] - cx[-1]) / 245)**2) * np.exp(-.5 * (cross / 130)**2)
    coastal_length = np.where(np.isin(rock, [1, 5, 6]), 14, 44 + 27 * soft_shelf)
    z = np.where(land, np.maximum(z, 22) * np.tanh(coast_km / coastal_length), z)
    ocean = -.15 - .002 * offshore_km
    bathy = np.minimum(-offshore_km * 2.2, np.maximum(z, -offshore_km * 10))
    return np.where(land, z, ocean), land, rock, contact, units["LT4"], bathy, lat, lon, extent


def erode(surface, land, rock, contact, red, pixel, steps):
    ny, nx = surface.shape
    fs.fastscape_init()
    try:
        fs.fastscape_set_nx_ny(nx, ny)
        fs.fastscape_setup()
        dx = pixel * 1000
        fs.fastscape_set_xl_yl((nx - 1) * dx, (ny - 1) * dx)
        fs.fastscape_set_bc(1111)
        ctx = fs.fastscapecontext
        ctx.dt = STEP_YEARS
        ctx.m, ctx.n, ctx.g1, ctx.g2 = .4, 1., 0., 0.
        ctx.kfsed, ctx.kdsed = -1., -1.
        ctx.kd = np.where(land.ravel(), .045, 0.)
        ks = np.array([unit[2] for unit in ROCKS])
        initial = surface.ravel()
        z = initial.copy()
        cell_area = land.ravel().astype(float) * dx**2
        row, col = np.indices(land.shape)
        row, col = row.ravel(), col.ravel()
        red_land = red & land
        for step in range(steps):
            bed_surface = ndi.gaussian_filter(z.reshape(land.shape), 4 / pixel)
            rock[red_land] = np.where(bed_surface[red_land] >= contact[red_land], 4, 3)
            ctx.kf = ks[rock.ravel()]
            _, rec, order = route(z.reshape(land.shape), land)
            area = cell_area.copy()
            _flow_accumulate_sd(area, order, rec)
            ctx.rec = rec.astype(np.int32) + 1
            ctx.stack = order.astype(np.int32) + 1
            ctx.length = np.maximum(np.hypot(row - row[rec], col - col[rec]), 1) * dx
            # 内核会除以汇水面积。海域无汇水贡献，给计算字段设置正下限。
            ctx.a = np.maximum(area, dx**2)
            ctx.h = z
            fs.streampowerlawsingleflowdirection()
            fs.diffusion()
            z = np.where(land.ravel(), np.maximum(ctx.h, .03), initial)
            assert np.isfinite(z).all(), "侵蚀步骤出现非有限高程"
            if (step + 1) % 8 == 0:
                print(f"区域侵蚀 {step + 1}/{steps}", flush=True)

    finally:
        fs.fastscape_destroy()
    return z.reshape(ny, nx)


def route(surface, land):
    """RichDEM 填洼和 D8 路由，返回条件化高程与下游优先的单向汇流图。"""
    nx = land.shape[1]
    raster = rd.rdarray(surface, no_data=-9999.)
    raster.geotransform = [0, 1, 0, 0, 0, -1]
    filled = rd.FillDepressions(raster, epsilon=True, in_place=False)
    props = rd.FlowProportions(filled, method="D8")[..., 1:]
    direction = np.where(props.max(axis=2) > 0, props.argmax(axis=2) + 1, 0)
    offset = np.array([0, -1, -nx - 1, -nx, -nx + 1, 1, nx + 1, nx, nx - 1])
    ids = np.arange(land.size).reshape(land.shape)
    rec = ids + offset[direction]
    # 区域边界和首次触海点都是出口。
    rec[~land] = ids[~land]
    rec[[0, -1], :] = ids[[0, -1], :]
    rec[:, [0, -1]] = ids[:, [0, -1]]
    rec = rec.ravel()
    filled = np.asarray(filled)
    order = np.argsort(filled.ravel())
    return filled, rec, order


@njit
def drainage_links(order, receiver, stream):
    """沿已排序的汇流图回溯每格的终点和下游首个河道像元。"""
    outlet = np.empty(receiver.size, np.int64)
    channel = np.full(receiver.size, -1, np.int64)
    for i in order:
        r = receiver[i]
        outlet[i] = i if r == i else outlet[r]
        if stream[i]:
            channel[i] = i
        elif r != i:
            channel[i] = channel[r]
    return outlet, channel


def analyze(design, pixel, steps):
    init, land, rock, contact, red, bathy, lat, lon, extent = source_surface(design, pixel)
    surface0 = erode(init, land, rock, contact, red, pixel, steps)
    coast0 = land.copy()
    # 抬升海面只淹没与外海连通的低谷，封闭洼地仍是陆地。
    sea = ndi.binary_propagation(~land, mask=(~land) | (surface0 < HIGHSTAND_M))
    drowned = land & sea
    land = ~sea
    bathy = np.where(drowned, surface0 - HIGHSTAND_M, bathy - HIGHSTAND_M)
    unconditioned = np.where(land, surface0 - HIGHSTAND_M, -.15 - .002 * ndi.distance_transform_edt(sea) * pixel)
    # 先打通局部洼地的出口，尽量减少靠填高谷底才能建立的流路。
    raster = rd.rdarray(unconditioned, no_data=-9999.)
    raster.geotransform = [0, 1, 0, 0, 0, -1]
    breached = np.asarray(rd.BreachDepressions(raster, in_place=False, topology="D8"))
    surface = np.where(land & (unconditioned > 0), np.maximum(breached, .002), unconditioned)
    bed_surface = ndi.gaussian_filter(surface + HIGHSTAND_M, 4 / pixel)
    rock[red & land] = np.where(bed_surface[red & land] >= contact[red & land], 4, 3)
    rock[sea] = 0
    grade, rec, order = route(surface, land)
    area = land.ravel().astype(float) * pixel**2
    _flow_accumulate_sd(area, order, rec)
    area = area.reshape(land.shape)
    n = land.size
    ids = np.arange(n)
    stream = land & (area >= STREAM_KM2)
    outlets, channels = drainage_links(order, rec, stream.ravel())
    outlets = outlets.reshape(land.shape)
    channel = channels.reshape(land.shape)
    nearest_channel = np.maximum(channel, 0)
    hand = np.maximum(0, surface - grade.ravel()[nearest_channel])
    sy, sx = np.gradient(ndi.gaussian_filter(surface, .7), pixel * 1000)
    slope = np.hypot(sx, sy)
    near_stream = ndi.distance_transform_edt(~stream) * pixel
    floodplain = land & (channel >= 0) & (surface < 550) & (hand < 14) & (near_stream < 6)
    floodplain &= (slope < .008) & (grade - surface < 5) & (area.ravel()[nearest_channel] >= 280)
    bedrock = rock.copy()
    rock[floodplain] = 7

    target = (np.abs(lon - EAST_ROUTE[-1][0]) < 2.5) & (np.abs(lat - EAST_ROUTE[-1][1]) < 2.9)
    shore = ~land & ndi.binary_dilation(land, structure=np.ones((3, 3)))
    east_mouths = np.flatnonzero((shore & target).ravel())
    # 优先查找真实从红层盆地集水、流向东岸陆架海的出口。
    red_outlets = np.bincount(outlets[red & land], minlength=n)
    primary = int(east_mouths[np.argmax(red_outlets[east_mouths])])
    basin = (outlets == primary) & land
    roots = rec == ids
    routed = land.ravel() & ~roots
    grades = grade.ravel()
    uphill = np.count_nonzero(grades[rec[routed]] >= grades[routed])
    displayed = stream.ravel() & routed
    adverse = surface.ravel()[rec[displayed]] - surface.ravel()[displayed]
    land_roots = roots.reshape(land.shape) & land
    interior_sinks = int(land_roots[1:-1, 1:-1].sum())
    total_area = float(land.sum() * pixel**2)
    collected = float(area.ravel()[roots].sum())
    assert uphill == 0, f"发现 {uphill} 个逆坡汇流连接"
    assert abs(collected - total_area) < 1e-5, (collected, total_area)
    assert interior_sinks == 0, f"仍有 {interior_sinks} 个内部封闭汇流点"
    assert red_outlets[primary] > 0 and area.ravel()[primary] >= MAIN_KM2, "盆地到东岸未形成贯通干流"
    primary_r, primary_c = np.unravel_index(primary, land.shape)

    def candidate(name, eligible, center, geology):
        # 按宿主、坡面和水系筛选；锚点只决定在同类露头里展示哪个候选。
        score = (lon - center[0])**2 * .71 + (lat - center[1])**2
        assert eligible.any(), name
        idx = int(np.argmin(np.where(eligible, score, np.inf)))
        return {"name": name, "index": idx, "lon": float(lon.flat[idx]), "lat": float(lat.flat[idx]),
                "height_m": round(float(surface.flat[idx]), 1), "basis": geology, "method_class": "user_override"}

    contact_km = ndi.distance_transform_edt(bedrock != 3) * pixel
    resources = [
        candidate("红砂岩采石候选", (rock == 4) & (slope > .0018) & (near_stream < 22),
                  (104.8, 31.15), "红层砂岩露头，邻近河谷，石材性质需区域设定"),
        candidate("制砖陶土候选", floodplain & basin & (area < STREAM_KM2) & (surface > 15),
                  (109.9, 32.8), "主流域缓坡冲积地，细泥含量为待确认设定"),
        candidate("灰岩石灰候选", (rock == 5) & (slope > .011) & (slope < .12) & (near_stream < 22),
                  (108.5, 27.4), "灰岩露头，邻近河谷，可作为石灰原料候选"),
        candidate("花岗岩采石候选", (rock == 6) & (slope > .012) & (slope < .15) & (near_stream < 22),
                  (119.1, 30.9), "花岗岩露头，邻近河谷，石材性质需区域设定"),
        candidate("层控铜矿设定点", (rock == 4) & (contact_km <= 12) & (slope > .0015) & (near_stream < 28),
                  (107.0, 30.4), "设定含铜流体沿砂岩运移，在局部富有机质的泥岩界面沉淀；品位与可采性未定"),
    ]
    metrics = {
        "name": "终图 v2 东方走廊区域样板", "source_sha256": hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        "projection": "LAEA", "center_lat_lon": CENTER, "extent_km": extent,
        "grid_yx": list(land.shape), "pixel_km": pixel, "seed": SEED, "steps": steps,
        "time_step_years": STEP_YEARS, "stream_power_m": .4, "stream_power_n": 1.,
        "regional_highstand_m": HIGHSTAND_M, "drowned_land_km2": float(drowned.sum() * pixel**2),
        "hill_diffusivity_m2_per_year": .045, "rock_k": {r[0]: r[2] for r in ROCKS},
        "land_km2": total_area, "drainage_area_balance_error_km2": collected - total_area,
        "uphill_links_on_conditioned_surface": int(uphill), "interior_sinks_on_conditioned_surface": interior_sinks,
        "land_area_draining_via_window_edge_km2": float(area[land_roots].sum()),
        "stream_threshold_km2": STREAM_KM2, "main_stream_threshold_km2": MAIN_KM2,
        "drawn_channel_cells": int(stream.sum()), "incision_depth_p95_m": round(float(np.percentile((init - surface0)[land], 95)), 1),
        "breach_over_5m_land_pct": round(float(np.mean((unconditioned - surface)[land] > 5) * 100), 2),
        "stream_steps_over_5m_uphill_before_fill": int(np.count_nonzero(adverse > 5)),
        "fill_over_5m_land_pct": round(float(np.mean((grade - surface)[land] > 5) * 100), 2),
        "fill_max_m": round(float(np.max((grade - surface)[land])), 1),
        "primary_estuary": {"lon": float(lon[primary_r, primary_c]), "lat": float(lat[primary_r, primary_c]),
                            "drainage_km2": float(area.ravel()[primary]),
                            "red_basin_contributing_km2": int(red_outlets[primary]) * pixel**2,
                            "red_basin_fraction": float(red_outlets[primary] / (red & land).sum())},
        "resources": resources,
        "layers": {
            "source_height": "visual_only", "coast_resampling": "interpolated", "regional_basin_gradient": "user_override",
            "regional_fold_hill_geometry": "user_override", "highstand_flooding": "parameterized_physics",
            "seeded_subgrid_initial_relief": "procedural_constrained", "stream_power_and_diffusion": "parameterized_physics",
            "surface_flow_and_drainage_area": "parameterized_physics", "bedrock_and_contacts": "user_override",
            "potential_alluvium": "procedural_constrained", "material_resource_candidates": "user_override",
        },
    }
    arrays = dict(init=init, land=land, coast0=coast0, bedrock=bedrock, rock=rock, surface=surface, grade=grade,
                  rec=rec, area=area, basin=basin, bathy=bathy, lat=lat, lon=lon, red=red,
                  contact=contact - HIGHSTAND_M)
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-km", type=int, choices=[2, 4], default=2)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/east_corridor.npz")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / "metrics.json"
    if args.render_existing:
        metrics = json.loads(meta.read_text(encoding="utf8"))
        arrays = np.load(args.data)
    else:
        start = time.perf_counter()
        design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf8")), set())
        metrics, arrays = analyze(design, args.pixel_km, args.steps)
        metrics["model_seconds"] = round(time.perf_counter() - start, 2)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.data, **arrays)
        meta.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    from design_east_corridor_figures import render

    render(metrics, arrays, args.out)


if __name__ == "__main__":
    main()
