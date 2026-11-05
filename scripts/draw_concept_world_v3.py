"""全新全球构图；海陆为人工设计，高程与生态配色仅为视觉示意。"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter,label
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap,LightSource
import cartopy.crs as ccrs
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"packages"))
from design_final_map import build,elevation,meander_polygon,feature_rng
from mythic.metrics.grid import LatLonGrid
from mythic.figures.style import setup_fonts
setup_fonts()
start=time.perf_counter()
out=ROOT/"figures/design/concepts/2026-11-05"
out.mkdir(parents=True,exist_ok=True)
d=json.loads((ROOT/"config/design/concept_v3.json").read_text(encoding="utf8"))
# 人工轮廓控制点的曲线插值，不声称过程生成海岸。
for group in ['masses','seas','plateaus','basins']:
    for item in d.get(group,[]):
        q=np.array(item['poly'],float)
        q=np.vstack([q,q[0]])
        delta=np.diff(q,axis=0)
        delta[:,0]*=np.cos(np.radians((q[1:,1]+q[:-1,1])/2))
        t=np.r_[0,np.cumsum(np.linalg.norm(delta,axis=1))]
        spline=CubicSpline(t,q,bc_type='periodic')
        item['poly']=spline(np.linspace(0,t[-1],max(180,len(q)*12),endpoint=False)).tolist()
for sea in d.get('seas',[]):
    sea['poly']=meander_polygon(sea['poly'],feature_rng(sea['id']),(45,18)).tolist()
g=LatLonGrid(2048,1024)
print("BUILD",flush=True)
b=build(d,g)
e=elevation(d,g,b)
land=b["land"]
LO,LA=np.meshgrid(g.lon,g.lat)
# 微地形仅服务于晕渲，不作为构造、侵蚀或水文过程的证据。
rng=np.random.default_rng(20261105)
fine=gaussian_filter(rng.standard_normal(g.shape),.8)
fine/=max(fine.std(),1e-9)
detail=np.clip((e-400)/2000,0,1)*(65+.045*np.maximum(e,0))*fine
relief=e+np.where(land,detail,0)
ocean=LinearSegmentedColormap.from_list("ocean",["#61bdb6","#237c9d","#154c73","#10324f","#0a233b"])
low=LinearSegmentedColormap.from_list("land",[(0,"#9fb487"),(.07,"#739b6b"),(.20,"#9ca679"),(.36,"#b9ab83"),(.55,"#ac977f"),(.72,"#bdb4a4"),(.85,"#e1dfd4"),(1,"#f7f7ef")])
rgb=ocean(np.clip(-e/4700,0,1))[...,:3]
lrgb=low(np.clip(relief/6900,0,1))[...,:3]
for zone in d.get("palette_zones",[]):
    x,y=zone["center"];rx,ry=zone["radii"]
    w=np.exp(-1.6*(((LO-x)/rx)**2+((LA-y)/ry)**2))
    w*=np.clip((3300-relief)/1800,0,1)
    col=np.array([.67,.48,.30]) if zone["kind"]=="dry" else np.array([.22,.43,.32])
    w*=.63 if zone["kind"]=="dry" else .38
    lrgb=lrgb*(1-w[...,None])+col*w[...,None]
# 高纬植被与雪仅是人工配色方案，不是气候分区。
north=np.clip((LA-47)/23,0,1)*np.clip((2300-relief)/1700,0,1)
lrgb=lrgb*(1-.28*north[...,None])+np.array([.38,.51,.43])*(.28*north[...,None])
snow=np.clip((relief-(5600-12*np.abs(LA)))/1100,0,1)
lrgb=lrgb*(1-snow[...,None])+np.array([.93,.94,.91])*snow[...,None]
shade=LightSource(azdeg=315,altdeg=43).hillshade(relief,vert_exag=16,dx=g.pixel_km_equator*1000,dy=g.pixel_km_equator*1000)
factor=.76+.38*shade
lrgb=np.clip(lrgb*factor[...,None],0,1)
rgb[land]=lrgb[land]
pol=b["polar"]
rgb[pol]=np.clip(np.array([.87,.93,.94])*factor[pol,None],0,1)
rgb[b["lake"]]=np.array([.23,.59,.68])
PC=ccrs.PlateCarree()
bg="#f1f0e8"
halo=[pe.withStroke(linewidth=2.8,foreground=bg,alpha=.8)]
def surface(ax,global_view=True):
    if global_view:ax.set_global()
    ax.set_facecolor("#0a233b")
    ax.imshow(rgb,extent=(-180,180,-90,90),origin="upper",transform=PC,regrid_shape=1400,interpolation="bilinear")
    ax.contour(g.lon,g.lat,land.astype(float),levels=[.5],colors="#183e48",linewidths=.27,transform=PC)
    ax.spines["geo"].set_edgecolor("#c0c9c5")
    ax.spines["geo"].set_linewidth(.65)
def footer(fig,extra=""):
    fig.text(.035,.025,"海岸：user_override  ·  地形晕渲与生态配色：visual_only  ·  可编辑构图提案，非演化或气候模拟"+extra,fontsize=8.8,color="#53676c")
fig=plt.figure(figsize=(24,13),facecolor=bg)
ax=fig.add_axes([.025,.095,.95,.78],projection=ccrs.EqualEarth())
surface(ax)
ax.gridlines(xlocs=range(-180,181,30),ylocs=range(-60,61,30),color="#88b0bb",alpha=.17,linewidth=.35)
for name,x,y,kind in d["labels"]:
    ax.text(x,y,name,transform=PC,color="#284647" if kind=="land" else "#d3e9e7",fontsize=11 if kind=="land" else 14,ha="center",path_effects=halo if kind=="land" else [pe.withStroke(linewidth=2,foreground="#194563",alpha=.8)])
fig.text(.04,.95,"群 海 之 冠",fontsize=30,color="#213d42")
fig.text(.04,.908,"全球构图提案 01  /  从霜冠峡湾到风暴群岛",fontsize=13,color="#5d7477")
fig.text(.96,.95,"MYTHIC PLANET LABORATORY",ha="right",fontsize=10,color="#5d7477")
fig.text(.96,.92,"2026.11.05  ·  地理设计草案",ha="right",fontsize=10,color="#5d7477")
footer(fig)
fig.savefig(out/"01_new_world_atlas.png",dpi=165,facecolor=bg)
plt.close(fig)
print("ATLAS_SAVED",flush=True)
fig=plt.figure(figsize=(22,12),facecolor=bg)
for pos,lon,lat,title in [([.035,.17,.45,.70],15,22,"内海半球"),([.515,.17,.45,.70],-165,0,"远洋半球")]:
    ax=fig.add_axes(pos,projection=ccrs.Orthographic(central_longitude=lon,central_latitude=lat))
    surface(ax)
    ax.set_title(title,fontsize=14,color="#39585e",pad=20)
fig.text(.04,.95,"同 一 颗 星 球",fontsize=27,color="#213d42")
fig.text(.04,.90,"同一份海岸与地形数据的球面视图",fontsize=12,color="#5d7477")
footer(fig)
fig.savefig(out/"02_planet_hemispheres.png",dpi=160,facecolor=bg)
plt.close(fig)
print("GLOBES_SAVED",flush=True)
fig=plt.figure(figsize=(20,12),facecolor=bg)
ax=fig.add_axes([.04,.12,.92,.74],projection=PC)
ax.set_extent([-68,58,8,61],crs=PC)
surface(ax,False)
for name,x,y,kind in d["labels"]:
    if -68<x<58 and 8<y<61:
        ax.text(x,y,name,transform=PC,color="#e5f1e8" if kind=="sea" else "#254b48",fontsize=15,ha="center",path_effects=[pe.withStroke(linewidth=2.5,foreground="#214c59" if kind=="sea" else bg)])
fig.text(.045,.95,"回 声 海  /  区 域 构 图",fontsize=25,color="#213d42")
fig.text(.045,.90,"分支内海、海心长岛、北向支海与东岸陆桥",fontsize=12,color="#5d7477")
footer(fig)
fig.savefig(out/"03_inland_sea_detail.png",dpi=160,facecolor=bg)
plt.close(fig)
# 保存可重建的设计格网；不冒充物理高程或气候结果。
dat=ROOT/"data/worlds/concept_v3"
dat.mkdir(parents=True,exist_ok=True)
ds=xr.Dataset({"elevation":(("lat","lon"),e.astype("float32"),{"units":"m","method_class":"visual_only","description":"人工地形示意，未作物理验证"}),"land":(("lat","lon"),land.astype("uint8"),{"method_class":"user_override"}),"lake_design":(("lat","lon"),b["lake"].astype("uint8"),{"method_class":"user_override","description":"人工湖泊位置草案，无水位平衡求解"})},coords={"lat":g.lat,"lon":g.lon},attrs={"method":"concept_v3_native_cartography","version":"2026-11-05","source":"config/design/concept_v3.json","stage_acceptance":0,"assumptions":"生态颜色为视觉配色，无气候解；河网未生成；海陆和高程为人工设计"})
ds.to_netcdf(dat/"concept_v3_2048x1024.nc")
labs,n=label(land)
areas=np.bincount(labs.ravel(),weights=np.broadcast_to(g.cell_area_km2,g.shape).ravel())
areas[0]=0
summary={"method":"concept_v3_native_cartography","method_class":{"coast":"user_override","elevation":"visual_only","palette":"visual_only"},"grid":[2048,1024],"ocean_fraction":float(1-(g.cell_area_km2*land).sum()/g.total_area_km2),"largest_land_component_share":float(areas.max()/areas.sum()),"elevation_visual_minmax_m":[float(e[land].min()),float(e[land].max())],"finite_elevation":bool(np.isfinite(e).all()),"staged_quality_metrics":"not_run_concept_not_stage_candidate","climate_and_hydrology":"not_solved","imagegen_attempt":"failed_network_error_no_image_returned","elapsed_s":time.perf_counter()-start,"files":{}}
for p in sorted(out.glob("*.png")):summary["files"][p.name]={"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
(dat/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
cards="".join('<section><h2>'+title+'</h2><a href="'+name+'"><img src="'+name+'" alt="'+title+'"></a></section>' for name,title in [("01_new_world_atlas.png","全球构图"),("02_planet_hemispheres.png","两个半球"),("03_inland_sea_detail.png","回声海区域")])
(out/"index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>新全球构图提案</title><style>body{background:#f1f0e8;color:#213d42;max-width:1500px;margin:40px auto;font-family:system-ui;padding:0 24px}img{width:100%;height:auto}h2{font-weight:500}p{line-height:1.8}</style><h1>群海之冠 · 全球构图提案</h1><p>2026-11-05。人工地理设计与视觉配色，不是演化、气候或水文模拟。新方案尚未替代旧终图，也未通过阶段验收。</p>'+cards+'</html>',encoding="utf8")
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
