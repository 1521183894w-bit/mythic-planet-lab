"""现世地形的回归检查。观感与阶段指标由标准图组独立评审。"""

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_present_terrain import CONFIG, DESIGN, axis, make_terrain  # noqa: E402
from design_final_map import apply_options  # noqa: E402
from mythic.metrics.grid import LatLonGrid  # noqa: E402
from mythic.metrics.spherical import distance_to_set_km  # noqa: E402


@pytest.fixture(scope="module")
def worlds():
    grid = LatLonGrid(512, 256)
    design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf-8")), set())
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = config["detail_seed"]
    return grid, [make_terrain(grid, design, config, s) for s in (seed, seed, seed + 1)]


def test_detail_seed_preserves_design(worlds):
    grid, (first, repeat, other) = worlds
    geo, bedrock, surface, _ = first
    for key in ("land", "polar", "lake", "salt", "shelf"):
        assert np.array_equal(geo[key], repeat[0][key])
        assert np.array_equal(geo[key], other[0][key])
    assert np.array_equal(bedrock, repeat[1])
    assert np.array_equal(surface, repeat[2])
    assert np.mean(np.abs(bedrock - other[1])) > 2
    ocean_fraction = np.sum(grid.cell_area_km2[~geo["land"]]) / grid.total_area_km2
    assert 0.705 < ocean_fraction < 0.725


def test_geographic_classes_and_relief(worlds):
    grid, (first, *_) = worlds
    geo, bedrock, surface, age = first
    land = geo["land"]
    ice = geo["polar"]
    assert np.isfinite(surface).all() and np.isfinite(bedrock).all()
    assert bedrock.min() > -12000 and surface.max() < 10000
    assert np.all(bedrock[~land] < 0)
    assert np.all(ice <= land) and np.all(geo["shelf"] <= ~land)
    assert np.all(surface[ice] >= bedrock[ice])
    assert np.array_equal(surface[~ice], bedrock[~ice])
    assert np.median(surface[ice] - bedrock[ice]) > 100
    dist = distance_to_set_km(grid, land, query=~land)
    open_water = (~land) & (dist > 520)
    young = open_water & (age < 12)
    old = open_water & (age > 90)
    assert np.count_nonzero(young) > 20 and np.count_nonzero(old) > 20
    assert np.median(bedrock[young]) > np.median(bedrock[old]) + 750
    assert np.percentile(bedrock[~land], .1) < -6200
    assert np.count_nonzero((bedrock > -200) & ~land) > 100
    rift = grid.rowcol(-2.4, 18.7)
    # 洼地允许是负高程陆地，不能用高程符号替代海陆图层。
    assert land[rift] and bedrock[rift] < 20


def test_axis_wraps_antimeridian():
    grid = LatLonGrid(1024, 512)
    mask = np.zeros(grid.shape, bool)
    ij = grid.rowcol([1.2, 1.2], [179.8, -179.8])
    mask[ij] = True
    _, s, _, d, length = axis(grid, [[178, 0], [-178, 0]], mask, 300)
    assert length == pytest.approx(445, abs=2)
    assert 0 < s.min() < s.max() < length
    assert abs(d[0] - d[1]) < 2
