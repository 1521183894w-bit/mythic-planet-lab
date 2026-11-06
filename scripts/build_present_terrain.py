"""从终图 v2 与现世构造假说重建陆海高程底图。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import xarray as xr
import yaml
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from design_final_map import apply_options, build, densify, elevation, feature_rng, lonlat_mask, meander, to_lonlat, unit, width_profile
from mythic.figures.netcdf import write_netcdf
from mythic.figures.world import code_version
from mythic.metrics.grid import LatLonGrid, chord_to_km, km_to_chord
from mythic.metrics.spherical import distance_to_set_km
from mythic.tectonics.crust import gdh1_depth_m

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/design/final_map_v2.yaml"
CONFIG = ROOT / "config/design/present_terrain_v1.yaml"
OUTPUT = ROOT / "data/worlds/present_v2/terrain_2048x1024.nc"
GDH1 = "Stein & Stein 1992, doi:10.1038/359123a0"
MARINE_REFERENCE = "Harris et al. 2014, doi:10.1016/j.margeo.2014.01.011"


def smoothstep(start, stop, values):
    t = np.clip((values - start) / (stop - start), 0, 1)
    return t * t * (3 - 2 * t)


def axis(grid, line, query, limit_km):
    """给走廊内的格点返回沿走向里程、有符号横向距离与端点距离。"""
    vertices, distance = densify(line, min(9, grid.pixel_km_equator / 2))
    tangent = np.gradient(vertices, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.cross(vertices, tangent)
    ij = np.flatnonzero(query)
    xyz = grid.xyz.reshape(-1, 3)[ij]
    chord, near = cKDTree(vertices).query(xyz, distance_upper_bound=km_to_chord(limit_km, grid.radius_km),
                                        workers=-1)
    keep = np.isfinite(chord)
    near = near[keep]
    xyz = xyz[keep]
    offset = np.sum((xyz - vertices[near]) * tangent[near], axis=1) * grid.radius_km
    along = (distance[near] * (grid.radius_km / 6371.0) + offset)
    across = np.arcsin(np.clip(np.sum(xyz * normal[near], axis=1), -1, 1)) * grid.radius_km
    return ij[keep], along, across, chord_to_km(chord[keep], grid.radius_km), distance[-1] * grid.radius_km / 6371.0


def trench_path(line):
    """在三维空间平滑控制轴，归一化回球面后用于出图与水深。"""
    nodes = unit(*np.asarray(line).T)
    s = np.r_[0, np.cumsum(chord_to_km(np.linalg.norm(np.diff(nodes, axis=0), axis=1)))]
    fine = np.linspace(0, s[-1], int(s[-1] / 25) + 2)
    lon, lat = to_lonlat(CubicSpline(s, nodes, bc_type="natural")(fine))
    return np.column_stack([lon, lat]).tolist()


def texture(rng, coordinates, scale):
    """在规则随机格点之间做三次样条插值。坐标单位 km，结果幅度限定在 [-1, 1]。"""
    xy = np.asarray(coordinates, dtype=np.float32) / np.float32(scale)
    origin = np.floor(xy.min(axis=1)) - 3
    xy -= origin[:, None]
    shape = np.ceil(xy.max(axis=1)).astype(int) + 4
    lattice = rng.uniform(-0.85, 0.85, size=tuple(shape)).astype(np.float32)
    return np.clip(map_coordinates(lattice, xy, order=3, mode="mirror"), -1, 1)


def aligned_relief(grid, land, spec, rng, old):
    width_max = float(np.max(spec["width_km"]))
    drift = 0.06 if spec.get("profile") == "rift" else 0.3
    line = meander(spec["line"], drift * width_max, feature_rng(spec["id"]))
    idx, s, x, d, length = axis(grid, line, land, 3 * width_max)
    if not idx.size:
        return idx, np.empty(0)
    # 横过山带的谷地在走向上密集、向两翼延伸。支脉在走向上更长，向外逐渐断续。
    cell = grid.pixel_km_equator
    warp = texture(rng, (s, x), max(4 * cell, 95))
    valley = texture(rng, (s + (65 + 0.35 * np.abs(x)) * warp, 0.39 * x), max(3.5 * cell, 67))
    ridge = texture(rng, (0.36 * s, x - 63 * warp), max(3.5 * cell, 72))
    rounded = texture(rng, (0.70 * s, x), max(5 * cell, 115))
    taper = smoothstep(-95, 125, s) * smoothstep(-95, 125, length - s)
    width = width_profile(spec["width_km"], np.clip(s, 0, length), length)
    shoulder = np.exp(-0.5 * (d / (0.72 * width)) ** 2)
    shoulder *= (1 - smoothstep(2 * width_max, 3 * width_max, d)) * taper
    is_rift = spec.get("profile") == "rift"
    if is_rift:
        shoulder *= smoothstep(0.09 * width, 0.27 * width, np.abs(x))
    if old:
        return idx, shoulder * (0.14 * spec["height_m"]) * (0.74 * rounded + 0.26 * ridge)
    core = np.exp(-(d / (0.38 * width)) ** 2)
    incision = (1 - np.abs(valley)) ** 6 - 0.24
    spurs = (0.56 * ridge + 0.44 * rounded)
    height = spec["height_m"]
    return idx, shoulder * height * ((0.29 - 0.19 * core) * spurs - (0.21 + 0.12 * core) * incision)


def land_relief(grid, geo, design, config, base, distance, seed):
    land = geo["land"]
    rng = np.random.default_rng(seed)
    xyz = grid.xyz[land].T * grid.radius_km
    coarse = texture(rng, xyz, max(6 * grid.pixel_km_equator, 132))
    fine = texture(rng, xyz, max(3.5 * grid.pixel_km_equator, 68))
    km = distance[land]
    h = base[land].copy()
    # 内陆高地、稳定地盾与沉积盆地各自拥有不同起伏幅度。
    amplitude = np.full(grid.shape, 65.0)
    for spec in design.get("lithology", ()):
        if spec["kind"] in {"craton", "granite", "sandstone"}:
            amplitude[lonlat_mask(grid, spec["poly"]) & land] = 190
    for spec in design.get("plateaus", ()):
        amplitude[lonlat_mask(grid, spec["poly"]) & land] = 300
    amplitude[~land] = 0
    # 岩性与地貌边界的随机起伏在格点尺度连续混合。
    categorical = amplitude.copy()
    amplitude *= 0.50
    for shift in (-2, -1, 1, 2):
        amplitude += 0.0625 * np.roll(categorical, shift, axis=1)
        vertical = np.clip(np.arange(grid.ny) + shift, 0, grid.ny - 1)
        amplitude += 0.0625 * categorical[vertical]
    h += amplitude[land] * (0.70 * coarse + 0.30 * fine)
    structured = np.zeros(grid.shape, np.float32).ravel()
    weight = np.zeros_like(structured)
    for spec in config["interior_provinces"]:
        width = spec["width_km"]
        idx, s, x, d, length = axis(grid, spec["line"], land, 2.5 * width)
        if not idx.size:
            continue
        env = np.exp(-(d / width) ** 2) * (1 - smoothstep(1.75 * width, 2.5 * width, d))
        # 低缓背斜与台地的基面由省份给定，丘陵、残丘细节沿构造分带。
        shape = texture(rng, (0.58 * s, x), max(4.5 * grid.pixel_km_equator, 95))
        dissection = texture(rng, (s, 0.55 * x), max(3.5 * grid.pixel_km_equator, 68))
        structured[idx] += env * spec["relief_m"] * (0.63 + 0.62 * shape - 0.19 * np.abs(dissection))
        weight[idx] += env
    old = set(config["old_mountains"])
    for spec in design["mountains"]:
        idx, local = aligned_relief(grid, land, spec, rng, spec["id"] in old)
        structured[idx] += local
    h += structured.reshape(grid.shape)[land] / np.maximum(1.0, weight.reshape(grid.shape)[land])
    # 沿岸低地平顺衔接浅海，岩岸和山前断崖可一直延伸至海面。
    near = 1 - smoothstep(6, 155, km)
    h -= near * np.minimum(np.maximum(h, 0) * 0.82, 118.0)
    # 锁定设计湖和海面以下的地堑；原设计湖高程暂作地表占位。
    locked = geo["lake"][land] | (base[land] <= 0)
    h = np.where(locked, base[land], np.maximum(2.0, h))
    result = base.copy()
    result[land] = h
    return result


def bathymetry(grid, geo, design, config, distance, seed):
    sea = ~geo["land"]
    dist = distance[sea]
    age = np.full(grid.shape, 165.0)
    fracture = np.zeros(grid.shape, np.float32).ravel()
    for ridge in config["ridges"]:
        for segment in ridge["segments"]:
            idx, _, _, d, _ = axis(grid, segment, sea, 165 * ridge["half_rate_km_myr"])
            age.flat[idx] = np.minimum(age.flat[idx], d / ridge["half_rate_km_myr"])
        for a, b in zip(ridge["segments"][:-1], ridge["segments"][1:]):
            if a[-1] == b[0]:
                continue
            idx, s, _, d, length = axis(grid, [a[-1], b[0]], sea, 170)
            # 转换断层两端接续轴线，外侧断裂带逐渐消失。
            taper = smoothstep(-80, 25, s) * smoothstep(-80, 25, length - s)
            fracture[idx] -= 470 * taper * np.exp(-(d / 42) ** 2)
    thermal = -gdh1_depth_m(age[sea])
    shelf_width = np.full(grid.shape, 119.0)
    # 年轻火山岛与岛弧的水下斜坡陡，陆架比老陆缘窄。
    arc_ids = [s["id"] for s in design["islands"] if s["type"] in {"arc", "hotspot"} and s["id"] != "R13"]
    oceanic_island = np.isin(geo["fid"], arc_ids)
    near_coast = sea & (distance < 480)
    island_distance = distance_to_set_km(grid, oceanic_island, query=near_coast, max_km=560)
    gap = island_distance[near_coast] - distance[near_coast]
    influence = 1 - smoothstep(10.0, 130.0, gap)
    shelf_width[near_coast] -= 93.0 * influence
    trench_depth = np.zeros(grid.shape, np.float32).ravel()
    for spec in config["trenches"]:
        idx, s, _, d, length = axis(grid, trench_path(spec["line"]), sea, 570)
        taper = smoothstep(-90, 300, s) * smoothstep(-90, 300, length - s)
        margin = taper * (1 - smoothstep(160, 570, d))
        shelf_width.flat[idx] = np.minimum(shelf_width.flat[idx], 119 + margin * (spec["shelf_km"] - 119))
        trench_depth[idx] -= spec["depth_anomaly_m"] * taper * np.exp(-(d / spec["width_km"]) ** 2)
    width = shelf_width[sea]
    shelf = -(8 + 186 * np.minimum(dist / width, 1.0) ** 1.45)
    slope_width = np.maximum(95, 90 + (width - 45) * 1.85)
    transition = smoothstep(width, width + slope_width, dist)
    rng = np.random.default_rng(seed)
    xyz = grid.xyz[sea].T * grid.radius_km
    coarse = texture(rng, xyz, max(5.0 * grid.pixel_km_equator, 108))
    fine = texture(rng, xyz, max(3.5 * grid.pixel_km_equator, 68))
    # 新生洋壳起伏比沉积覆盖的老洋盆强，近陆缘的沉积减弱洋底起伏。
    roughness = (105 + 380 * np.exp(-age[sea] / 46)) * smoothstep(width, width + 410, dist)
    floor = thermal + roughness * (0.68 * coarse + 0.32 * fine)
    floor += fracture.reshape(grid.shape)[sea] + trench_depth.reshape(grid.shape)[sea]
    result = np.zeros(grid.shape)
    result[sea] = shelf * (1 - transition) + floor * transition

    # 已有热点岛链向海下延伸。环礁与未出露海山采用不同的顶深。
    peaks = []
    for i, symbol in enumerate(geo["symbols"]):
        kind = symbol["kind"]
        if kind in {"atoll", "lone_atoll", "seamount", "volcanic"}:
            frng = feature_rng(f"{symbol['id']}-{i}")
            summit = -frng.uniform(900, 2100) if kind == "seamount" else -frng.uniform(30, 85)
            peaks.append((symbol["lon"], symbol["lat"], frng.uniform(27, 62), summit))
    xyz_all = grid.xyz.reshape(-1, 3)
    points = cKDTree(xyz_all)
    for lon, lat, radius, summit in peaks:
        center = unit(lon, lat)
        ids = np.asarray(points.query_ball_point(center, km_to_chord(3.6 * radius, grid.radius_km)), dtype=int)
        ids = ids[sea.flat[ids]]
        chord = np.linalg.norm(xyz_all[ids] - center, axis=-1)
        d = chord_to_km(chord, grid.radius_km) / radius
        z = result.flat[ids]
        result.flat[ids] += np.maximum(0, summit - z) * np.exp(-d ** 1.45) * (1 - smoothstep(2.7, 3.6, d))


    # 命名浅湾按中心轴深度控制，避免宽阔的陆表海自动变成外洋深海。
    for gulf in design["gulfs"]:
        depth = config["shallow_gulfs"].get(gulf["id"])
        if depth is None:
            continue
        w = float(np.max(gulf["width_km"]))
        idx, s, x, d, length = axis(grid, gulf["line"], sea, w)
        local_w = width_profile(gulf["width_km"], np.clip(s, 0, length), length)
        gate = smoothstep(-0.18 * w, 0.26 * w, s) * (1 - smoothstep(0.40 * local_w, 0.67 * local_w, d))
        z = -depth * smoothstep(0, 0.31 * local_w, distance.flat[idx])
        result.flat[idx] += gate * np.maximum(z - result.flat[idx], 0)

    # 内海、弧后海和裂谷海的局部基面。地峡、海峡和岛屿保持设计边界。
    for spec in config["marine_basins"]:
        w = spec["width_km"]
        idx, s, _, d, length = axis(grid, spec["line"], sea, 2.6 * w)
        gate = (1 - smoothstep(1.5 * w, 2.6 * w, d))
        gate *= smoothstep(-0.35 * w, 0.75 * w, s) * smoothstep(-0.35 * w, 0.75 * w, length - s)
        z = -spec["depth_m"] * np.exp(-0.5 * (d / w) ** 2) * smoothstep(0, 0.72 * w, distance.flat[idx])
        result.flat[idx] = (1 - gate) * result.flat[idx] + gate * z
    result[sea] = np.minimum(result[sea], -2.0)
    return result, age


def make_terrain(grid, design, config, seed):
    # 终图海陆使用独立的固定种子。改变细节种子只改变地形。
    geo = build(design, grid, seed=1)
    land = geo["land"]
    reference = elevation(design, grid, geo, ice_surface=False)
    dl = distance_to_set_km(grid, ~land, query=land)
    ds = distance_to_set_km(grid, land, query=~land)
    bedrock = land_relief(grid, geo, design, config, reference, dl, seed)
    bathy, age = bathymetry(grid, geo, design, config, ds, seed)
    bedrock[~land] = bathy[~land]
    ice = geo["polar"]
    surface = bedrock.copy()
    surface[ice] = np.maximum(bedrock[ice], 3300 * (1 - np.exp(-dl[ice] / 450)) + 0.19 * bedrock[ice])
    return geo, bedrock, surface, age


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(512, 1024, 2048), default=2048)
    parser.add_argument("--seed", type=int, help="细节种子，默认使用配置值")
    parser.add_argument("--output", type=Path, help="默认在 data/worlds/present_v2 下按格网命名")
    args = parser.parse_args()
    design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf-8")), set())
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    grid = LatLonGrid(args.width, args.width // 2, config["planet"]["radius_km"])
    output = args.output or OUTPUT.with_name(f"terrain_{grid.name}.nc")
    seed = config["detail_seed"] if args.seed is None else args.seed
    t = perf_counter()
    geo, bedrock, surface, age = make_terrain(grid, design, config, seed)
    dims = ("lat", "lon")
    land = geo["land"]
    def layer(array, units, kind, method, assumptions, reference="config/design/final_map_v2.yaml", dtype="float32"):
        attrs = {"units": units, "method_class": kind, "method": method, "resolution": grid.name,
                 "assumptions": assumptions, "reference": reference}
        return dims, array.astype(dtype), attrs

    lith = np.zeros(grid.shape, np.uint8)
    for i, spec in enumerate(design["lithology"], 1):
        lith[lonlat_mask(grid, spec["poly"]) & land] = i
    water = np.zeros(grid.shape, np.uint8)
    water[~land] = 1
    water[geo["sea_masks"]["G15"] & ~land] = 3
    water[geo["lake"]] = 2
    water[geo["salt"]] = 3
    water[geo["shelf"]] = 4
    sea = ~land
    ds = xr.Dataset({
        "elevation": layer(surface, "m", "procedural_constrained", "present terrain v1",
                           "冰盖区为设计冰面，其余为陆面、湖区地表占位和洋底", GDH1 + "；" + MARINE_REFERENCE),
        "bedrock_elevation": layer(bedrock, "m", "procedural_constrained", "present terrain v1",
                                   "相对现今海平面；设计湖区仍用地表占位，未计算湖底", GDH1 + "；" + MARINE_REFERENCE),
        "land": layer(land, "1", "user_override", "design_final_map.build v2", "保留 v2 指标口径",
                      dtype="uint8"),
        "water_type": layer(water, "1", "user_override", "design_final_map.build v2",
                            "0 干地，1 海水，2 设计淡水湖，3 设计咸水内流湖，4 浮冰架", dtype="uint8"),
        "grounded_ice": layer(geo["polar"], "1", "user_override", "design_final_map.build v2",
                              "接地冰覆盖范围由终图给定", dtype="uint8"),
        "lithology_province": layer(lith, "1", "user_override", "design_final_map.lonlat_mask v2",
                                  "0 未指定；其他值为岩性设计假说编号", dtype="uint8"),
        "open_ocean_reference_age": layer(np.where(sea, age, np.nan), "Myr", "procedural_constrained",
                                         "ridge distance / half-spreading rate v1",
                                         "洋盆冷却深度代理量，局部内海和陆架不代表真实洋壳年龄", GDH1),
    }, coords={"lat": ("lat", grid.lat, {"units": "degrees_north"}),
               "lon": ("lon", grid.lon, {"units": "degrees_east"})},
        attrs={"world_id": "present_v2", "display_name": "终图 v2 · 现世地形",
               "source": "config/design/final_map_v2.yaml + config/design/present_terrain_v1.yaml",
               "design_sha256": sha256(DESIGN.read_bytes()).hexdigest(),
               "terrain_config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
               "seed": str(seed), "method_class_elevation": "procedural_constrained", "code_version": code_version(),
               "radius_km": grid.radius_km, "planet": json.dumps(config["planet"]),
               "lithology_codes": json.dumps({i: s["id"] for i, s in enumerate(design["lithology"], 1)}),
               "grid": grid.name})
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=output.name, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    for name in ds.data_vars:
        ds[name].encoding.update(zlib=True, complevel=3)
    try:
        write_netcdf(ds, temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{grid.name} · {perf_counter() - t:.1f} 秒 · {output} · {output.stat().st_size / 2**20:.2f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
