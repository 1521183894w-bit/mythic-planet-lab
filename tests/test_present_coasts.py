"""现世岸段候选的球面几何、拓扑和远岸回归检查。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mythic.metrics.grid import LatLonGrid  # noqa: E402
from present_coasts import corridor, make_coast  # noqa: E402


def test_coast_preserves_locked_features_and_offshore_relief():
    grid = LatLonGrid(720, 360)
    land = np.broadcast_to(grid.lon[None, :] < 0, grid.shape).copy()
    elev = np.where(land, 200.0, -230.0)
    empty = np.zeros(grid.shape, bool)
    lake = empty.copy()
    lake[grid.rowcol(.5, -2.5)] = True
    t1 = empty.copy()
    t1[grid.rowcol(8, 1)] = True
    geo = {"land": land, "lake": lake, "shelf": empty, "polar": empty,
           "fid": np.full(grid.shape, "", dtype="U2"),
           "sea_masks": dict.fromkeys(["T1", "T2", "G7", "G23", "G15"], empty)}
    geo["sea_masks"]["T1"] = t1
    coast = {"regions": [{"id": "sample", "kind": "fault",
                          "cuts": [{"line": [[2, 0], [-2, 0], [-6, .4]], "width": [300, 260], "depth": 190},
                                   {"line": [[-21, 0], [-24, 2]], "width": 270, "depth": 160}],
                          "headlands": [{"line": [[-2, 8], [1, 8], [3, 7.8]],
                                         "width": [330, 270], "height": 260}]}]}
    revised, bedrock, surface = make_coast(grid, geo, elev, elev, coast)
    new_land = revised["land"]
    assert new_land[lake].all() and not new_land[t1].any()
    assert new_land[grid.rowcol(1, -22)]  # 孤立的槽形不挖成湖
    assert not new_land[grid.rowcol(0, -4)]
    assert new_land[grid.rowcol(7.8, 2.3)]
    assert np.all(bedrock[~new_land] < 0)
    far = np.abs(grid.lat) > 15
    assert np.array_equal(surface[far], elev[far])
    assert np.array_equal(new_land[far], land[far])
    repeated = make_coast(grid, geo, elev, elev, coast)
    assert np.array_equal(repeated[0]["land"], new_land)
    assert np.array_equal(repeated[1], bedrock)


def test_coast_corridor_crosses_longitude_seam():
    grid = LatLonGrid(720, 360)
    spec = {"line": [[178, 1], [-178, 1]], "width": [250, 175], "skew": .2}
    idx, shape = corridor(grid, spec, "same-rift", "glacial")
    i, j = np.unravel_index(idx[shape > .15], grid.shape)
    assert j.min() == 0 and j.max() == grid.nx - 1
    assert np.all(np.abs(grid.lat[i] - 1) < 2)
