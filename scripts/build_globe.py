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

from design_final_map import LatLonGrid, apply_options, build, elevation
from mythic.figures.style import shaded_elevation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config/design/final_map_v2.yaml"
VIEWER = ROOT / "viewer"
DEFAULT_OUTPUT = ROOT / "figures/design/v2/globe/index.html"


def png_url(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def surfaces(grid, design):
    geography = build(design, grid)
    land = geography["land"]
    elev = elevation(design, grid, geography)
    shaded = shaded_elevation(elev, grid.pixel_km_equator, vert_exag=3, land=land)[..., :3]
    # 沿用终图的示意阴影与陆地色阶，统一远洋色调，避免把设色解读成植被或气候。
    sea = np.clip((elev + 6500) / 6500, 0, 1)[..., None] ** 2.1
    ocean_rgb = np.array([19, 40, 59]) + sea * np.array([45, 63, 74])
    terrain = 255 * (.63 * shaded + .17 * shaded.mean(axis=2, keepdims=True)) + [37, 33, 27]
    relief = np.where(land[..., None], terrain, ocean_rgb)
    relief[geography["polar"]] = .20 * relief[geography["polar"]] + .80 * np.array([212, 225, 229])
    coast = np.empty((*grid.shape, 3), dtype=np.uint8)
    coast[:] = [30, 59, 77]
    coast[land] = [189, 176, 137]
    for key, color in {"shelf": [180, 203, 215], "lake": [73, 123, 150], "salt": [98, 151, 150]}.items():
        relief[geography[key]] = color
        coast[geography[key]] = color
    return {"relief": png_url(np.rint(relief).astype(np.uint8)), "coast": png_url(coast)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = SOURCE.read_bytes()
    design = apply_options(yaml.safe_load(source.decode("utf-8")), set())
    grid = LatLonGrid(2048, 1024)
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
        "textures": surfaces(grid, design),
    }
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
