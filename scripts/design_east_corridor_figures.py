"""绘制东方走廊样板的地形、水系、岩性与资源图。"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy import ndimage as ndi

from design_east_corridor import CENTER, ROCKS, ROOT
from design_civilization_zones import EAST_ROUTE
from mythic.figures.regional import laea_forward, laea_window, draw_graticule
from mythic.figures.style import add_scalebar, setup_fonts

PAPER = "#faf8f1"
INK = "#223e47"
MUTED = "#567078"
BLUE = "#227a9f"
RIVER = "#1b597e"
GOLD = "#995a28"
HALO = [pe.withStroke(linewidth=3.5, foreground=PAPER)]
MARGIN = 90
TRANSECT = [(95.8, 34.6), (101.5, 32.4), *EAST_ROUTE, (125.4, 34.8)]
ETOPO = ROOT / "data/earth_reference/raw/etopo2022/ETOPO_2022_v1_60s_N90W180_surface.nc"


def terrain_rgb(z, bathy, land, pixel):
    km = np.maximum(0, z) / 1000
    ramp = LinearSegmentedColormap.from_list("terrain_east", [
        (0, "#9bad82"), (.055, "#b9bd8a"), (.15, "#b3a574"),
        (.35, "#9c866c"), (.62, "#b8a89b"), (.78, "#d9cbc0"), (1, "#fcf2df"),
    ])
    rgb = ramp(np.clip(km / 5.5, 0, 1))[..., :3]
    light = LightSource(azdeg=322, altdeg=42)
    h = light.hillshade(np.where(land, z, np.maximum(bathy, -60)), vert_exag=3.2,
                        dx=pixel * 1000, dy=pixel * 1000, fraction=1)
    land_rgb = np.clip(rgb * (.74 + h[..., None] * .39), 0, 1)
    sea = LinearSegmentedColormap.from_list("shelf_east", ["#deedf0", "#b7d9e1", "#85b8c9"])
    sea_rgb = sea(np.clip(np.sqrt(np.maximum(0, -bathy) / 1700), 0, 1))[..., :3]
    return np.where(land[..., None], land_rgb, sea_rgb)


def coordinates(metrics, shape):
    x0, _, _, y1 = metrics["extent_km"]
    dy = metrics["pixel_km"]
    return x0 + (np.arange(shape[1]) + .5) * dy, y1 - (np.arange(shape[0]) + .5) * dy


def project(points):
    lon, lat = np.array(points).T
    return np.column_stack(laea_forward(*CENTER, lat, lon))


def format_axis(ax, extent):
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#a1b1af")
        spine.set_linewidth(.7)


def text_geo(ax, lon, lat, name, color=INK, **kwargs):
    x, y = project([(lon, lat)])[0]
    ax.text(x, y, name, ha="center", va="center", color=color, weight="bold",
            path_effects=HALO, zorder=10, **kwargs)


def rivers(ax, x, y, data, extent, threshold, primary=None):
    land, area, rec = (data[k] for k in ("land", "area", "rec"))
    win = ((x >= extent[0] - 6) & (x <= extent[1] + 6))[None, :]
    win = win & (((y >= extent[2] - 6) & (y <= extent[3] + 6))[:, None])
    ids = np.flatnonzero(land & (area >= threshold) & win)
    ids = ids[rec[ids] != ids]
    downstream = rec[ids]
    previous = np.full(land.size, -1, dtype=np.int32)
    previous[downstream] = ids
    simple = (np.bincount(downstream, minlength=land.size) == 1) & (previous >= 0)
    visible = np.zeros(land.size, dtype=bool)
    visible[ids] = True

    def points(nodes):
        row, col = np.unravel_index(nodes, land.shape)
        xy = np.column_stack([x[col], y[row]])
        # 仅移动一进一出的中间节点；河口和汇合点保持原位置。
        mask = simple[nodes] & visible[nodes]
        p, q = np.unravel_index(previous[nodes[mask]], land.shape)
        u, v = np.unravel_index(rec[nodes[mask]], land.shape)
        xy[mask] = .6 * xy[mask] + .2 * np.column_stack([x[q] + x[v], y[p] + y[u]])
        return xy

    segments = np.stack([points(ids), points(downstream)], axis=1)
    bigger = np.clip(np.log10(area.flat[ids] / threshold + 1), 0, 2.9)
    ax.add_collection(LineCollection(segments, colors=BLUE, linewidths=.34 + bigger * .32,
                                     alpha=.9, capstyle="round", zorder=4))
    if primary is not None:
        trunk = primary.ravel()[ids] & (area.flat[ids] >= 4000)
        ax.add_collection(LineCollection(segments[trunk], colors=RIVER,
                                         linewidths=.7 + bigger[trunk] * .38,
                                         capstyle="round", zorder=5))


def titles(fig, label, title, subtitle):
    fig.text(.042, .958, label, color="#aa6434", fontsize=11, weight="bold", va="top")
    fig.text(.041, .924, title, color=INK, fontsize=25, weight="bold", va="top")
    fig.text(.042, .872, subtitle, color=MUTED, fontsize=10.5, va="top")


def profile_line(data, x, y):
    xy = project(TRANSECT)
    dense = np.concatenate([np.linspace(a, b, max(4, int(np.linalg.norm(b - a) / 2)), endpoint=False)
                            for a, b in zip(xy, xy[1:])] + [xy[-1:]])
    step = x[1] - x[0]
    indices = np.array([(y[0] - dense[:, 1]) / step, (dense[:, 0] - x[0]) / step])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    samples = {k: ndi.map_coordinates(data[k], indices, order=0 if k in {"land", "rock", "bedrock"} else 1)
               for k in ("land", "surface", "bathy", "rock", "bedrock", "contact")}
    return xy, distance, samples


def map_extent(full):
    return (full[0] + MARGIN, full[1] - MARGIN, full[2] + MARGIN, full[3] - MARGIN)


def render(metrics, arrays, out):
    setup_fonts()
    data = {k: arrays[k] for k in arrays}
    land, z = data["land"], data["surface"]
    pix = metrics["pixel_km"]
    x, y = coordinates(metrics, land.shape)
    full = tuple(metrics["extent_km"])
    overview = map_extent(full)
    color = terrain_rgb(z, data["bathy"], land, pix)
    mouth = metrics["primary_estuary"]
    mouth_xy = project([(mouth["lon"], mouth["lat"])])[0]
    sites = metrics["resources"]
    transect, distance, section = profile_line(data, x, y)
    # A、B 放大窗直接在全区格网上截取，保留同一条水系。
    basin_xy = project([(105.55, 31.35)])[0]
    boxes = [
        (basin_xy[0] - 245, basin_xy[0] + 245, basin_xy[1] - 190, basin_xy[1] + 190),
        (mouth_xy[0] - 310, mouth_xy[0] + 180, mouth_xy[1] - 190, mouth_xy[1] + 190),
    ]
    rock_colors = [r[1] for r in ROCKS]
    rock_map = ListedColormap(rock_colors)

    def paint(ax, extent, geology=False, threshold=580):
        if geology:
            ax.imshow(data["rock"], cmap=rock_map, vmin=-.5, vmax=len(ROCKS) - .5,
                      extent=full, interpolation="nearest")
        else:
            ax.imshow(color, extent=full, interpolation="nearest")
        ax.contour(x, y, land, levels=[.5], colors=["#4f7984"], linewidths=.47, zorder=3)
        rivers(ax, x, y, data, extent, threshold, data["basin"])
        format_axis(ax, extent)

    fig = plt.figure(figsize=(16.8, 13.5), facecolor=PAPER)
    titles(fig, "区域样板  /  01", "东方走廊", f"东方大高原东麓 → 红层盆地 → 东岸陆架海    ·    地形与地表水系    ·    格网 {pix} km")
    ax = fig.add_axes([.041, .134, .918, .706])
    paint(ax, overview)
    draw_graticule(ax, CENTER, data["lat"], data["lon"], overview, step=5)
    ax.contour(x, y, data["basin"], levels=[.5], colors=[GOLD], linewidths=.68,
               linestyles="--", alpha=.85, zorder=6)
    for name, (a, b, c, d) in zip(("A", "B"), boxes):
        ax.add_patch(Rectangle((a, c), b - a, d - c, fc="none", ec=PAPER, lw=1.7, zorder=8))
        ax.text(a + 14, d - 18, name, va="top", color=PAPER, fontsize=11, weight="bold",
                bbox=dict(facecolor=INK, edgecolor="none", pad=3.5), zorder=9)
    text_geo(ax, 96.3, 35.4, "东方大高原东麓", fontsize=12)
    text_geo(ax, 105.1, 31.8, "红层盆地", fontsize=12)
    text_geo(ax, 110.1, 26.0, "南方喀斯特高原", fontsize=11)
    text_geo(ax, 124.9, 38.4, "东岸陆架海", color=RIVER, fontsize=12)
    ax.scatter(*mouth_xy, marker="D", s=50, facecolor="#fff2cf", edgecolor=RIVER, linewidth=1.5, zorder=9)
    ax.annotate("入海口", xy=mouth_xy, xytext=(mouth_xy[0] + 110, mouth_xy[1] - 70),
                color=RIVER, fontsize=10, weight="bold", path_effects=HALO,
                arrowprops=dict(arrowstyle="-", color=RIVER, lw=.8), zorder=9)
    add_scalebar(ax, overview, 250, color=INK)
    legend = [Line2D([0], [0], color=RIVER, lw=2, label="主要河道"),
              Line2D([0], [0], color=BLUE, lw=.85, label="较小支流"),
              Line2D([0], [0], color=GOLD, lw=1.1, ls="--", label="东方主流域分水岭"),
              Line2D([0], [0], marker="D", mfc="#fff2cf", mec=RIVER, color="none", label="主流域入海口")]
    fig.legend(handles=legend, ncol=4, loc="lower left", bbox_to_anchor=(.036, .084),
               frameon=False, labelcolor=INK, fontsize=10)
    fig.text(.042, .063, f"标出的东方主流域约 {mouth['drainage_km2']:,.0f} km²    ·    A、B 对应 03 号图的盆地与河口放大窗。",
             color=INK, fontsize=11, weight="bold")
    fig.text(.042, .035, "终图大格局与岩性锚点为设计。河谷使用参数化侵蚀，水系按地形汇流。边缘汇水限于区域计算窗。",
             color=MUTED, fontsize=9.5)
    fig.savefig(out / "01_terrain_rivers.png", dpi=174, facecolor=PAPER)
    plt.close(fig)

    fig = plt.figure(figsize=(16.8, 14.5), facecolor=PAPER)
    titles(fig, "区域样板  /  02", "从红砂岩到海岸冲积地", "区域岩性、材料候选点与东西剖面    ·    资源点按宿主岩和出露环境选取")
    ax = fig.add_axes([.041, .327, .918, .524])
    paint(ax, overview, geology=True, threshold=1500)
    ax.plot(*transect.T, color=INK, lw=1.0, ls="--", path_effects=HALO, zorder=7)
    for label, point in zip(("P", "P′"), transect[[0, -1]]):
        ax.text(*point, label, fontsize=11, weight="bold", color=INK, path_effects=HALO, zorder=8)
    for j, site in enumerate(sites, 1):
        px, py = project([(site["lon"], site["lat"])])[0]
        ax.scatter(px, py, marker="o", s=110, color=INK, edgecolor=PAPER, lw=1, zorder=9)
        ax.text(px, py, str(j), color=PAPER, ha="center", va="center", fontsize=8, weight="bold", zorder=10)
    add_scalebar(ax, overview, 250, color=INK)
    fig.legend(handles=[Patch(facecolor=r[1], edgecolor="#7c8d86", linewidth=.35, label=r[0]) for r in ROCKS[1:]],
               ncol=4, loc="lower left", bbox_to_anchor=(.037, .282), frameon=False, labelcolor=INK, fontsize=9.5)
    fig.text(.042, .266, "    ".join(f"{i}  {s['name']}" for i, s in enumerate(sites, 1)),
             color=INK, fontsize=10.4, weight="bold")
    sec = fig.add_axes([.072, .092, .87, .144], facecolor="#f1eee5")
    upper = np.where(section["land"], section["surface"], section["bathy"]) / 1000
    is_red = np.isin(section["bedrock"], [3, 4])
    is_cover = np.isin(section["bedrock"], [2, 3, 4, 5, 7])
    red_body = ndi.gaussian_filter1d(is_red.astype(float), 11)
    cover = ndi.gaussian_filter1d(is_cover.astype(float), 12)
    base = upper - cover * (.23 + .96 * red_body)
    mid = np.minimum(upper, np.maximum(base, section["contact"] / 1000))
    sec.fill_between(distance, -1.65, upper, color=rock_colors[1])
    sec.fill_between(distance, base, upper, color=rock_colors[2])
    sec.fill_between(distance, base, mid, where=red_body > .22, color=rock_colors[3], interpolate=True)
    sec.fill_between(distance, mid, upper, where=red_body > .22, color=rock_colors[4], interpolate=True)
    sec.fill_between(distance, upper, 0, where=~section["land"], color=rock_colors[0], interpolate=True)
    for code, code_color in enumerate(rock_colors[1:], 1):
        sec.fill_between(distance, upper - .09, upper, where=section["rock"] == code,
                         color=code_color, interpolate=True, zorder=4)
    sec.plot(distance, upper, color=INK, linewidth=.65, zorder=5)
    sec.axhline(0, color="#517883", alpha=.55, lw=.6)
    sec.set_ylim(-1.55, max(5.1, upper.max() + .35))
    sec.set_xlim(distance[0], distance[-1])
    sec.set_ylabel("高程（km）", fontsize=9, color=INK)
    sec.set_xlabel("沿 P–P′ 距离（km）", fontsize=9, color=INK, labelpad=3)
    sec.tick_params(colors=MUTED, labelsize=8)
    for spine in sec.spines.values():
        spine.set_color("#a1b1af")
    for name, xfrac in [("高原东麓", .05), ("红层沉积充填", .47), ("冲积低地", .69), ("陆架海", .95)]:
        sec.text(xfrac, .92, name, transform=sec.transAxes, fontsize=9.3, color=INK,
                 va="top", ha="center", path_effects=HALO)
    fig.text(.042, .042, "剖面地表取自同一格网，地下接触与层厚为设计示意，纵向夸张。铜矿点为成矿设定，开采条件待定。",
             fontsize=9.3, color=MUTED)
    fig.savefig(out / "02_geology_resources.png", dpi=174, facecolor=PAPER)
    plt.close(fig)

    fig = plt.figure(figsize=(16.8, 8.9), facecolor=PAPER)
    titles(fig, "区域样板  /  03", "从山前支流到主河口", "两块同尺度窗口，单图约 490 × 380 km。小河道按相同汇流阈值显示。")
    for j, (extent, title) in enumerate(zip(boxes, ("A  红层盆地", "B  东岸入海口"))):
        ax = fig.add_axes([.041 + j * .47, .175, .449, .645])
        paint(ax, extent, threshold=metrics["stream_threshold_km2"])
        ax.contour(x, y, data["rock"] == 7, levels=[.5], colors=["#ebd9a5"],
                   linewidths=.45, alpha=.9, zorder=3)
        ax.set_title(title, fontsize=14, weight="bold", color=INK, loc="left", pad=9)
        add_scalebar(ax, extent, 50, color=INK)
        if j == 1:
            ax.contour(x, y, data["coast0"], levels=[.5], colors=[GOLD],
                       linewidths=.8, linestyles="--", alpha=.85, zorder=5)
            ax.scatter(*mouth_xy, marker="D", s=60, facecolor="#fff2cf", edgecolor=RIVER, linewidth=1.5, zorder=9)
            ax.annotate("主河入海", xy=mouth_xy, xytext=mouth_xy + [58, 35], color=INK, fontsize=10,
                        weight="bold", path_effects=HALO,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=.8), zorder=9)
        else:
            text_geo(ax, 105.75, 32.15, "红砂岩与泥岩", fontsize=10)
    fig.text(.043, .118, "浅金细线圈出候选冲积地。B 图的棕色虚线为海侵前岸线。",
             color=INK, fontsize=10.7, weight="bold")
    fig.text(.043, .075, f"每格 {pix} km    ·    支流显示阈值 {metrics['stream_threshold_km2']} km²    ·    海岸以终图为基线，叠加 {metrics['regional_highstand_m']} m 海侵情景。",
             color=MUTED, fontsize=9.7)
    fig.savefig(out / "03_rivers_detail.png", dpi=187, facecolor=PAPER)
    plt.close(fig)

    if ETOPO.exists():
        compare_earth(out, pix, color, full, boxes[1], mouth_xy)


def compare_earth(out, pixel, color, full, coast_extent, mouth_xy):
    """对照地图共享等面积投影、像元、着色和窗宽，观察岸线与近岸起伏。"""
    from netCDF4 import Dataset

    center = (29.25, 121.7)
    lat, lon, bounds = laea_window(*center, 490, pixel)
    with Dataset(ETOPO) as ds:
        slat = np.asarray(ds["lat"][:])
        slon = np.asarray(ds["lon"][:])
        ii = np.flatnonzero((slat > lat.min() - .06) & (slat < lat.max() + .06))
        jj = np.flatnonzero((slon > lon.min() - .06) & (slon < lon.max() + .06))
        patch = np.asarray(ds["z"][ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1], dtype=float)
    y = (lat - slat[ii[0]]) / (slat[1] - slat[0])
    x = (lon - slon[jj[0]]) / (slon[1] - slon[0])
    earth = ndi.map_coordinates(patch, [y, x], order=1)
    ref_extent = (-245, 245, -190, 190)

    fig = plt.figure(figsize=(16.8, 9), facecolor=PAPER)
    titles(fig, "区域样板  /  04", "同尺度看东岸", "每窗 490 × 380 km，使用相同的高程设色与阴影。")
    for j, (rgb, image_extent, window, title) in enumerate([
        (color, full, coast_extent, "东方走廊  ·  陆架海西岸"),
        (terrain_rgb(earth, earth, earth > 0, pixel), bounds, ref_extent, "地球  ·  浙江沿海"),
    ]):
        ax = fig.add_axes([.041 + j * .47, .163, .449, .66])
        ax.imshow(rgb, extent=image_extent, interpolation="nearest")
        format_axis(ax, window)
        add_scalebar(ax, window, 50, color=INK)
        ax.set_title(title, fontsize=14, weight="bold", color=INK, loc="left", pad=9)
        if j == 0:
            ax.scatter(*mouth_xy, s=38, marker="D", color="#fff2cf", edgecolor=RIVER, lw=1.1, zorder=9)
    fig.text(.043, .104, "地球参考  NOAA ETOPO 2022  60 角秒地形与水深，重采样到相同像元。此图只比较地表形态。",
             color=MUTED, fontsize=10)
    fig.savefig(out / "04_earth_coast_comparison.png", dpi=165, facecolor=PAPER)
    plt.close(fig)
