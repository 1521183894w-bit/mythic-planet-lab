"""在终图 v2 上定位候选文明腹地，并用示意高程核对东方出海通道。"""
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
import numpy as np
from PIL import Image
import yaml

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from design_final_map import LatLonGrid, apply_options, build, elevation
from mythic.figures.style import setup_fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "design" / "v2"
SOURCE = OUT / "01_final_map_coordinates.png"
DESIGN = ROOT / "config" / "design" / "final_map_v2.yaml"
INK = "#182d40"
WEST, INLAND, EAST = "#ffcb65", "#60e1e5", "#eaa6fa"
EAST_INK = "#782a74"
EAST_ROUTE = [(106.1, 31.4), (109, 31.6), (112, 32.3), (115, 33.2), (118, 34.2), (121.55, 34.54)]
EAST_BOUNDS = (95, 127, 24, 42)
PORTS = [
    (-35.77, 50.54, WEST), (-26.81, 51.94, WEST), (-21.01, 55.81, WEST),
    (3.96, 42.98, INLAND), (15.03, 27.16, INLAND), (*EAST_ROUTE[-1], EAST),
]
WARM_CURRENT = [(-78, 20), (-75, 33), (-67, 42), (-54, 48), (-45, 50), (-42.4, 51.5)]
WARM_BRANCH = [(-45, 50), (-43.5, 55), (-37, 59), (-31, 59.5), (-25.7, 58.4)]


def oval(lon, lat, width, height):
    angle = np.linspace(0, 2 * np.pi, 121)
    return np.column_stack((lon + width / 2 * np.cos(angle), lat + height / 2 * np.sin(angle)))


def overview():
    meta = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf8"))
    width, height = meta["canvas_pixels"]
    px, py, pw, ph = meta["axes_position"]
    bottom, first, last, total = 230, 132, 1580, 1880
    with Image.open(SOURCE) as base:
        img = np.array(base.convert("RGB").crop((0, first, width, last)))
    dpi = 150
    fig = plt.figure(figsize=(width / dpi, total / dpi), dpi=dpi, facecolor="white")
    fig.figimage(img, xo=0, yo=bottom, origin="upper", zorder=-10)
    proj = ccrs.EqualEarth(central_longitude=meta["central_longitude_deg"])
    geographic = ccrs.PlateCarree()
    y_offset = bottom + last - height
    ax = fig.add_axes([px, (py * height + y_offset) / total, pw, ph * height / total], projection=proj)
    ax.set_global()
    ax.set_axis_off()

    # 上缘经度刻度单独画，给讨论图标题留出干净的页眉。
    for lon in range(-150, 181, 30):
        label = f"{abs(lon)}°" + ("W" if lon < 0 else "E" if 0 < lon < 180 else "")
        xy = proj.transform_point(max(lon, -149.999999), 90, geographic)
        ax.annotate(label, xy=xy, xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, weight="bold", color="#173b58",
                    annotation_clip=False, clip_on=False)

    def line(points, color, width=2, style="-", arrow=False):
        xy = np.asarray(points)
        ax.plot(*xy.T, transform=geographic, color=color, linewidth=width, linestyle=style,
                path_effects=[pe.Stroke(linewidth=width + 1.4, foreground=INK), pe.Normal()], zorder=5)
        if arrow:
            start, end = proj.transform_points(geographic, *xy[-2:].T)[:, :2]
            ax.annotate("", xy=end, xytext=start, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=width,
                                        mutation_scale=15, shrinkA=0, shrinkB=0,
                                        path_effects=[pe.Stroke(linewidth=width + 1.4, foreground=INK), pe.Normal()]))

    for lon, lat, w, h, color in [
        (-22.6, 51.4, 14, 9.2, WEST),
        (5.5, 42.8, 9.8, 5.3, INLAND), (12.4, 27.7, 10.5, 5.7, INLAND),
        (110.4, 33.6, 15.8, 7.3, EAST),
    ]:
        line(oval(lon, lat, w, h), color, width=1.65)

    line(WARM_CURRENT, WEST, width=2.7, arrow=True)
    line(WARM_BRANCH, WEST, width=2.4, arrow=True)
    line([(-24.0, 36.15), (-19.6, 36.05), (-15.4, 36.1), (-10.6, 36.2)],
         INLAND, width=2, style="--")
    line(EAST_ROUTE, EAST, width=2.25, style="--")
    for lon, lat, color in PORTS:
        ax.scatter(lon, lat, transform=geographic, marker="D", s=70,
                   facecolor=color, edgecolor=INK, linewidth=1.25, zorder=8)

    def tag(text, target, label, color):
        xy = proj.transform_point(*target, geographic)
        xytext = proj.transform_point(*label, geographic)
        ax.annotate(text, xy=xy, xytext=xytext, ha="center", va="center",
                    fontsize=13, weight="bold", color=color, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.4", fc=INK, ec=color, lw=1.1),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.6,
                                    path_effects=[pe.Stroke(linewidth=3.0, foreground=INK), pe.Normal()]))

    tag("① 西北海贸带", (-43.4, 51.3), (-67, 59), WEST)
    tag("② 主内海两岸", (-19.4, 36.1), (-46.5, 28), INLAND)
    tag("③ 东方出海走廊", (121.55, 34.54), (155, 39), EAST)

    fig.text(0.028, 0.972, "终图 v2  |  文明舞台讨论图", fontsize=25, weight="bold", color=INK, va="top")
    fig.text(0.029, 0.932, "北纬 25°–55° 为主要故事带，西峡湾海是向北延伸的候选海贸节点。",
             fontsize=11.5, color="#455c6d", va="top")
    for xpos, color, title, body in [
        (.030, WEST, "① 西北海贸带", "西方大岛和洲 A 西岸联港\n金色箭头为候选暖流，北支通向西峡湾海外"),
        (.358, INLAND, "② 主内海两岸", "西北湾与内海南湾各留一片候选腹地\n青色虚线标出连通西大洋的西口"),
        (.685, EAST, "③ 东方出海走廊", "红层盆地向东北接东岸陆架海\n紫色虚线为陆路选线，详见东方放大图"),
    ]:
        fig.add_artist(Line2D([xpos, xpos + .283], [.105, .105], color=color, lw=3, transform=fig.transFigure))
        fig.text(xpos, .088, title, color=INK, fontsize=14, weight="bold", va="top")
        fig.text(xpos, .063, body, color="#3b5365", fontsize=10, linespacing=1.45, va="top")
    fig.text(.029, .015, "◇ 候选港口    彩色圈为候选腹地范围    |    标注为设计示意（visual_only）",
             color="#526776", fontsize=9, va="bottom")
    fig.savefig(OUT / "07_civilization_zones.png", dpi=dpi)
    plt.close(fig)


def eastern_detail(design, grid, land, elev):
    x, y = grid.lon, grid.lat[::-1]
    land, elev = land[::-1], elev[::-1]
    west, east, south, north = EAST_BOUNDS
    ix, iy = (x >= west) & (x <= east), (y >= south) & (y <= north)
    el = elev[np.ix_(iy, ix)]
    mask = land[np.ix_(iy, ix)]
    xp, yp = x[ix], y[iy]
    # 把候选折线逐段采样到设计格网，用来核对是否有越海或跨高地。
    samples = np.concatenate([np.linspace(a, b, 120) for a, b in zip(EAST_ROUTE, EAST_ROUTE[1:])])
    i = np.abs(y[:, None] - samples[:, 1]).argmin(axis=0)
    j = np.abs(x[:, None] - samples[:, 0]).argmin(axis=0)
    route_z = elev[i, j]
    assert land[i, j].all() and route_z.max() < 500

    # 在候选港口经线量取包含港口的连续低地南北断面。
    lon, lat = EAST_ROUTE[-1]
    j = np.abs(x - lon).argmin()
    i = np.abs(y - lat).argmin()
    low = land[:, j] & (elev[:, j] < 500)
    stops = np.flatnonzero(~low)
    lo, hi = stops[stops < i][-1] + 1, stops[stops > i][0] - 1
    span = (hi - lo + 1) * np.deg2rad(y[1] - y[0]) * grid.radius_km
    span_rounded = round(span / 50) * 50
    print(f"候选线全为陆地，高程约 {route_z.min():.0f}–{route_z.max():.0f} m")
    print(f"{x[j]:.2f}°E 连续 <500 m 低地 {y[lo]:.2f}–{y[hi]:.2f}°N，约 {span_rounded} km")

    fig = plt.figure(figsize=(13.5, 8), facecolor="#fbfaf6")
    fig.text(.058, .954, "东方放大  |  红层盆地到东岸陆架海", color=INK, fontsize=23, weight="bold", va="top")
    fig.text(.06, .905, "按终图示意高程分色，检查低地是否连通。", color="#455c6d", fontsize=12, va="top")
    ax = fig.add_axes([.06, .16, .88, .69])
    colors = ["#b9d8e6", "#c1dba1", "#d5ce90", "#cfab7c", "#b68774", "#f1e4d2"]
    levels = [-1, 0, 500, 1000, 2000, 3500, 10000]
    ax.pcolormesh(xp, yp, np.where(mask, np.maximum(el, 0), -.5), shading="auto",
                  cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)), rasterized=True)
    ax.contour(xp, yp, mask, levels=[.5], colors=["#55727c"], linewidths=.7)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / np.cos(np.deg2rad(33)))
    xt, yt = np.arange(95, 127, 5), np.arange(24, 43, 3)
    ax.set_xticks(xt, [f"{v}°E" for v in xt])
    ax.set_yticks(yt, [f"{v}°N" for v in yt])
    ax.grid(color="#56717a", alpha=.24, linestyle=":", linewidth=.65)
    ax.tick_params(colors="#466170", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#a8b6ba")

    basin = next(f for f in design["basins"] if f["id"] == "B6")
    polygon = np.array(basin["poly"] + basin["poly"][:1])
    ax.plot(*polygon.T, color=EAST_INK, lw=1.3, ls=":", zorder=5)
    halo = [pe.withStroke(linewidth=4, foreground="#fbfaf6")]
    ax.plot(*np.array(EAST_ROUTE).T, color=EAST_INK, lw=2.65, ls="--",
            path_effects=halo, zorder=7)
    ax.scatter(lon, lat, s=85, marker="D", color=EAST_INK, edgecolor="#fbfaf6", lw=1.6, zorder=8)
    ax.annotate("", xy=(x[j], y[hi]), xytext=(x[j], y[lo]), zorder=6,
                arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, color="#263d4a", lw=1.6, path_effects=halo))
    for x0, y0, label in [
        (96.5, 35.6, "东方大高原"), (106.1, 30.25, "红层盆地"),
        (112.5, 34.2, "候选低地通道"), (109.2, 25.6, "南方喀斯特高原"),
        (124.0, 35.0, "东岸陆架海"),
    ]:
        ax.text(x0, y0, label, color=INK, fontsize=12, weight="bold", ha="center",
                path_effects=halo, zorder=8)
    ax.annotate(f"临海收窄段\n低地南北约 {span_rounded} km", xy=(x[j], 35.6), xytext=(121.2, 39.7),
                ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=9,
                bbox=dict(boxstyle="round,pad=.42", fc="#fbfaf6", ec="#647f8d"),
                arrowprops=dict(arrowstyle="-", lw=1.1, color="#647f8d"))

    legend = [Line2D([0], [0], color=c, lw=8, label=label) for c, label in zip(
        colors, ["水域", "<500 米", "500–1000 米", "1000–2000 米", "2000–3500 米", "≥3500 米"])]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .086),
               ncol=6, frameon=False, fontsize=10)
    fig.text(.06, .050, "紫色点线为盆地设计范围，虚线为候选陆路，菱形为候选港口。",
             color="#455c6d", fontsize=10)
    fig.text(.06, .025, "高程为终图示意（visual_only），格网间距约 16–20 km。局部坡度、河流和港湾深度需另行核对。",
             color="#455c6d", fontsize=10)
    fig.savefig(OUT / "08_east_lowland_corridor.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    setup_fonts()
    design = apply_options(yaml.safe_load(DESIGN.read_text(encoding="utf8")), set())
    grid = LatLonGrid(2048, 1024)
    geography = build(design, grid)
    elev = elevation(design, grid, geography)
    overview()
    eastern_detail(design, grid, geography["land"], elev)


if __name__ == "__main__":
    main()
