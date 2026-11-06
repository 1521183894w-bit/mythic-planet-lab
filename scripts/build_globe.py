"""从原终图 v2 导出可离线打开的单页地球仪。"""

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from build_present_terrain import COASTS, CONFIG as TERRAIN_CONFIG
from design_final_map import LatLonGrid, apply_options, build, elevation
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import shaded_elevation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/design/final_map_v2.yaml"
VIEWER = ROOT / "viewer"
DEFAULT_OUTPUT = ROOT / "figures/design/v2/globe/index.html"


def png_url(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def surfaces(grid, geography, elev, terrain=False):
    land = geography["land"]
    # 经度环绕，阴影在 180° 也有两侧样本。
    z = np.pad(elev, ((0, 0), (1, 1)), mode="wrap")
    mask = np.pad(land, ((0, 0), (1, 1)), mode="wrap")
    shaded = shaded_elevation(z, grid.pixel_km_equator, vert_exag=3, land=mask)[:, 1:-1, :3]
    # 沿用终图的示意阴影与陆地色阶，统一远洋色调，避免把设色解读成植被或气候。
    sea = np.clip((elev + 6500) / 6500, 0, 1)[..., None] ** 2.1
    ocean_rgb = np.array([19, 40, 59]) + sea * np.array([45, 63, 74])
    land_rgb = 255 * (.63 * shaded + .17 * shaded.mean(axis=2, keepdims=True)) + [37, 33, 27]
    relief = np.where(land[..., None], land_rgb, ocean_rgb)
    relief[geography["polar"]] = .20 * relief[geography["polar"]] + .80 * np.array([212, 225, 229])
    coast = np.empty((*grid.shape, 3), dtype=np.uint8)
    coast[:] = [30, 59, 77]
    coast[land] = [189, 176, 137]
    for key, color in {"shelf": [180, 203, 215], "lake": [73, 123, 150], "salt": [98, 151, 150]}.items():
        relief[geography[key]] = color
        coast[geography[key]] = color
    result = {"relief": png_url(np.rint(relief).astype(np.uint8)), "coast": png_url(coast)}
    if terrain:
        bathy = np.where(land[..., None], [111, 106, 91], 255 * shaded)
        bathy[geography["shelf"]] = [180, 203, 215]
        bathy[geography["salt"]] = [98, 151, 150]
        result["bathy"] = png_url(np.rint(bathy).astype(np.uint8))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--terrain", type=Path, help="使用阶段 1 的现世高程 NetCDF")
    parser.add_argument("--coasts", type=Path, default=COASTS, help="用于核对岸段来源的设计 YAML")
    args = parser.parse_args()

    source = SOURCE.read_bytes()
    design = apply_options(yaml.safe_load(source.decode("utf-8")), set())
    if args.terrain:
        with open_netcdf(args.terrain) as ds:
            if (ds.attrs["design_sha256"] != hashlib.sha256(source).hexdigest() or
                    ds.attrs["terrain_config_sha256"] != hashlib.sha256(TERRAIN_CONFIG.read_bytes()).hexdigest()):
                raise ValueError("高程文件与当前设计或地形配置不一致，请重建高程")
            coast_hash = ds.attrs.get("coast_config_sha256")
            if coast_hash and coast_hash != hashlib.sha256(args.coasts.read_bytes()).hexdigest():
                raise ValueError("岸段设计配置已变更，请重建高程")
            coast_meta = dict(ds.land.attrs)
            elev = ds.elevation.values
            grid = LatLonGrid.from_shape(elev.shape, float(ds.attrs["radius_km"]))
            if not (np.array_equal(ds.lat, grid.lat) and np.array_equal(ds.lon, grid.lon)):
                raise ValueError("高程文件的球面像元坐标不匹配")
            water = ds.water_type.values
            geography = {"land": ds.land.values.astype(bool), "polar": ds.grounded_ice.values.astype(bool),
                         "lake": water == 2, "salt": water == 3, "shelf": water == 4}
            terrain_meta = dict(ds.elevation.attrs)
            terrain_meta["seed"] = ds.attrs["seed"]
            terrain_meta["name"] = "现世地形"
    else:
        grid = LatLonGrid(2048, 1024)
        geography = build(design, grid)
        elev = elevation(design, grid, geography)
    manifest = {
        "schema": 1,
        "world": "终图 v2",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "grid": [grid.nx, grid.ny],
        "radius_km": grid.radius_km,
        "projection": "equirectangular; north first; cell centered; longitude -180 to +180",
        "layers": {
            "relief": {"name": "示意地形", "method_class": "visual_only", "units": "m（源格网）",
                       "method": "design_final_map.elevation v2 + shaded_elevation"},
            "coast": {"name": "海陆轮廓", "method_class": "user_override", "units": "类别",
                      "method": "design_final_map.build v2"},
        },
        "textures": surfaces(grid, geography, elev, terrain=args.terrain is not None),
    }
    if args.terrain:
        manifest["layers"]["relief"] = terrain_meta
        manifest["layers"]["coast"] = {**coast_meta, "name": "海陆轮廓"}
        manifest["layers"]["bathy"] = {**terrain_meta, "name": "海底高程"}
        # 16 位带偏移整数，误差不超过 0.5 m。蓝通道保留地表类别。
        packed = (np.rint(elev).astype(np.int32) + 12000).astype(np.uint16)
        manifest["sample"] = png_url(np.stack([packed >> 8, packed & 255, water], axis=-1).astype(np.uint8))
        manifest["sample_offset_m"] = 12000
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    for slot, content in {
        "STYLE": (VIEWER / "globe.css").read_text(encoding="utf-8"),
        "APP": (VIEWER / "globe.js").read_text(encoding="utf-8"),
        "WORLD": json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
    }.items():
        html = html.replace(f"<!-- {slot} -->", content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{args.output.resolve()} · {grid.name} · {args.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
