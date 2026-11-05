"""在原终图v2上叠加投影一致的经纬网与刻度；不重建、不修改地理。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"figures/design/v2/01_final_map.png"
out=ROOT/"figures/design/v2/01_final_map_coordinates.png"
img=np.array(Image.open(source).convert("RGB"))
assert img.shape[:2]==(1650,3000),"底图尺寸不符合原始20×11英寸、150dpi版式"
fig=plt.figure(figsize=(20,11),dpi=150,facecolor="white")
fig.figimage(img,xo=0,yo=0,origin="upper",zorder=-10)
# 版式、投影和中央经线直接沿用scripts/design_final_map.py的原图。
proj=ccrs.EqualEarth(central_longitude=30)
pc=ccrs.PlateCarree()
ax=fig.add_axes([.02,.07,.96,.84],projection=proj)
ax.set_global()
ax.set_facecolor("none")
ax.patch.set_alpha(0)
ax.spines["geo"].set_visible(False)
gl=ax.gridlines(crs=pc,draw_labels={"bottom":"x","top":"x"},
    xlocs=np.arange(-180,181,30),ylocs=np.arange(-75,76,15),
    x_inline=False,y_inline=False,rotate_labels=False,
    linewidth=.55,color="#334f65",alpha=.48,linestyle="--")
gl.xformatter=LongitudeFormatter(number_format=".0f",dateline_direction_label=False)
gl.yformatter=LatitudeFormatter(number_format=".0f")
gl.xlabel_style={"size":10,"weight":"bold","color":"#173b58",
                 "path_effects":[pe.withStroke(linewidth=2.5,foreground="white")]}
gl.ylabel_style={"size":10,"weight":"bold","color":"#173b58",
                 "path_effects":[pe.withStroke(linewidth=2.5,foreground="white")]}
gl.xpadding=7
gl.ypadding=7
# 弯曲边框上逐一放置纬度，避免自动标注只留下赤道附近几个刻度。
for lat in range(-90,91,15):
    x,y=proj.transform_point(-149.999999,lat,pc)
    text="0°" if lat==0 else f"{abs(lat)}°{'N' if lat>0 else 'S'}"
    for side in (-1,1):
        ax.annotate(text,xy=(side*abs(x),y),xycoords="data",
                    xytext=(side*7,0),textcoords="offset points",
                    ha="left" if side>0 else "right",va="center",
                    fontsize=10,weight="bold",color="#173b58",clip_on=False,
                    path_effects=[pe.withStroke(linewidth=2.5,foreground="white")])
ax.plot(np.linspace(-180,180,721),np.zeros(721),transform=pc,color="#234863",lw=.8,alpha=.5)
ax.plot(np.zeros(361),np.linspace(-90,90,361),transform=pc,color="#234863",lw=.8,alpha=.5)
fig.savefig(out,dpi=150,facecolor="white")
fig.canvas.draw()
meta={"method":"native_cartographic_coordinate_overlay","method_class":"visual_only",
      "base_image":source.relative_to(ROOT).as_posix(),
      "base_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
      "projection":"EqualEarth","central_longitude_deg":30,
      "longitude_interval_deg":30,"latitude_interval_deg":15,
      "canvas_pixels":[3000,1650],"axes_position":[.02,.07,.96,.84],
      "geography_changed":False,"output":out.relative_to(ROOT).as_posix()}
(out.with_suffix(".json")).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8",newline="\n")
plt.close(fig)
print(json.dumps(meta,ensure_ascii=False,indent=2))
