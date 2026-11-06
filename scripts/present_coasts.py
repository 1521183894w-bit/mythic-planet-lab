"""现世岸线的区域成因假说和岸段重建。"""

from __future__ import annotations

import numpy as np

from design_final_map import feature_rng, lonlat_mask, unit, width_profile
from mythic.metrics.grid import km_to_chord
from mythic.metrics.spherical import distance_to_set_km
from mythic.metrics.topology import label


def corridor(grid, spec, feature, kind):
    """在球面计算非对称槽谷或岩脊的无量纲横向轮廓。"""
    from build_present_terrain import axis, texture, trench_path

    width = np.max(spec["width"])
    line = spec["line"]
    start = unit(*line[0])
    reach = max(np.linalg.norm(unit(*point) - start) for point in line)
    reach += km_to_chord(width * 1.1 + 80, grid.radius_km)
    # 粗包围盒包含跨接缝的轴线，精确距离与方向在球面上计算。
    xyz = grid.xyz
    query = np.ones(grid.shape, dtype=bool)
    for component in range(3):
        query &= np.abs(xyz[..., component] - start[component]) < reach
    if len(line) > 2:
        line = trench_path(line)
    idx, s, cross, distance, length = axis(grid, line, query, 0.85 * width)
    if not idx.size:
        return idx, np.empty(0)
    w = width_profile(spec["width"], np.clip(s, 0, length), length) / 2
    rng = feature_rng(feature)
    resistance = texture(rng, (s, cross), max(4 * grid.pixel_km_equator, 92))
    # 侧壁随岩体差异起伏，扰动只在设计走廊内，受槽宽和地貌类别限幅。
    contrast = {"glacial": 0.29, "fault": 0.26, "drowned": 0.16}[kind]
    w *= 1 + spec.get("skew", 0) * np.sign(cross) + contrast * resistance
    # 窄端在向外伸出的方向更尖灭，宽口保留水下肩部。
    cap = np.maximum(np.maximum(-s, s - length), 0)
    r = np.hypot(distance, 0.6 * cap) / w
    return idx, np.clip(1 - r * r, 0, 1)


def make_coast(grid, geo, bedrock, surface, design):
    """以同一走向轮廓决定岸形和邻岸高程。远离岸段的原 DEM 保持逐格一致。"""
    from build_present_terrain import smoothstep

    old_land = geo["land"]
    locked = geo["lake"] | geo["shelf"] | (geo["fid"] == "I4")
    for key in ("T1", "T2", "G7", "G23", "G15"):
        locked |= geo["sea_masks"][key]
    lowered = np.zeros(grid.shape, np.float32)
    raised = np.zeros(grid.shape, np.float32)
    cut_id = np.zeros(grid.shape, np.uint8)
    raise_id = np.zeros(grid.shape, np.uint8)
    polar_raises = np.zeros(grid.shape, bool)
    for region_id, region in enumerate(design["regions"], 1):
        for i, spec in enumerate(region.get("cuts", ())):
            idx, shape = corridor(grid, spec, f"{region['id']}-cut-{i}", region["kind"])
            # 冰蚀槽侧更陡，浅溺谷和陆架海的水下剖面较平缓。
            exponent = {"glacial": 0.72, "fault": 0.94, "drowned": 1.42}[region["kind"]]
            z = spec["depth"] * shape ** exponent
            better = z > lowered.flat[idx]
            lowered.flat[idx[better]] = z[better]
            cut_id.flat[idx[better]] = region_id
        for i, spec in enumerate(region.get("headlands", ())):
            idx, shape = corridor(grid, spec, f"{region['id']}-rock-{i}", region["kind"])
            z = spec["height"] * shape
            better = z > raised.flat[idx]
            raised.flat[idx[better]] = z[better]
            raise_id.flat[idx[better]] = region_id
            if region["id"] in {"PW", "PE"}:
                polar_raises.flat[idx[better]] = True
        for spec in region.get("islands", ()):
            poly = lonlat_mask(grid, spec["poly"])
            if not poly.any():
                continue
            depth = distance_to_set_km(grid, ~poly, query=poly)[poly]
            z = spec["height"] * smoothstep(0, 85, depth)
            better = z > raised[poly]
            idx = np.flatnonzero(poly)[better]
            raised.flat[idx] = z[better]
            raise_id.flat[idx] = region_id
            if region["id"] in {"PW", "PE"}:
                polar_raises.flat[idx] = True

    cut = (lowered > 0) & ~locked
    # 旧距岸剖面只是粗略基准。只有显式洋盆深槽的深度量级屏蔽新岸。
    rise = (raised > 0) & ~cut & ~locked & (old_land | (bedrock > -6000))
    land = (old_land | rise) & ~cut
    # 由切槽生成的水格必须与既有外海或原内海连续。去掉孤立的试挖凹地。
    labels, _ = label(~land)
    ocean_ids = np.unique(labels[(~old_land) & (~geo["sea_masks"]["G15"]) & (~land)])
    cut &= np.isin(labels, ocean_ids)
    land = (old_land | rise) & ~cut

    new_z = bedrock.copy()
    new_z[cut] = np.minimum(-lowered[cut], bedrock[cut])
    new_z[rise] = np.maximum(raised[rise], bedrock[rise])
    changed = land ^ old_land
    if changed.any():
        # 在改岸边界周围将旧陆坡和陆地基面并入新岸。过渡随实际距离渐消。
        change_distance = distance_to_set_km(grid, changed, max_km=145)
        near = (change_distance < 145) & (~locked)
        new_distance = distance_to_set_km(grid, ~land, query=near & land, max_km=160)
        wet_distance = distance_to_set_km(grid, land, query=near & ~land, max_km=180)
        blend = 1 - smoothstep(10, 145, change_distance)
        flank = near & land & (~rise)
        new_z[flank] -= blend[flank] * np.maximum(0, new_z[flank] - (25 + 6 * new_distance[flank]))
        shelf = near & ~land & (~cut) & (bedrock > -3300)
        new_z[shelf] += blend[shelf] * np.maximum(0, -(15 + 4.0 * wet_distance[shelf]) - new_z[shelf])

    region_field = np.zeros(grid.shape, np.uint8)
    region_field[cut] = cut_id[cut]
    region_field[rise] = raise_id[rise]
    polar = (geo["polar"] | (rise & polar_raises)) & land
    new_surface = np.where(polar, np.maximum(surface, new_z), new_z)
    # 冰面与非冰面未受沿岸调整的格点保持原值。
    out = {**geo, "land": land, "polar": polar, "coast_design_region": region_field}
    return out, new_z, new_surface
