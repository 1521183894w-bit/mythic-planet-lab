"""U44固定I4海道断面、分区及真实补壳事件；不把像元宽当真实水道最窄宽。"""
import numpy as np
import matplotlib.pyplot as plt
from mythic.metrics.topology import label,component_sums
from mythic.tectonics.pocket_exchange_spike import thin_partition,covered_boundary_ports
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.geometry import xyz_to_lonlat
import pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def add_evidence(summ,native,runs,z,g,out,save):
    land={};labs={}
    # 原生输出land的定义和标准指标一致；不从探针厚度推断整个岛。
    for k in ['control','uniform','pockets']:
        case=summ['cases'][k]
        # 输出land与高程正负在此候选一致，另在下方断言。
        from netCDF4 import Dataset
        name={'control':'control','uniform':'uniform','pockets':'pockets'}[k];c='u44_partition_'+name
        p=ROOT/f'data/worlds/design_v2/candidates/{c}/design_v2_forward_v2_0Ma_{c}_1024x512.nc'
        with Dataset(p) as ds:land[k]=np.array(ds.variables['land'][:],bool)
        assert np.array_equal(land[k],z[k]>0)
        labs[k],_=label(land[k])
    iy=int(np.argmin(abs(g.lat-38)));ix=int(np.argmin(abs(g.lon-5)))
    island=labs['control'][iy,ix]
    rows=np.flatnonzero(labs['control'][:,ix]==island)
    north=int(rows.min())
    lab,n=label(land['control']);sizes=component_sums(lab,n,g.cell_area_km2)
    main=int(np.argmax(sizes[1:]))+1
    mainrows=np.flatnonzero((lab[:,ix]==main)&(np.arange(len(g.lat))<north))
    south=int(mainrows.max())
    channel=np.arange(south+1,north)
    section=dict(lon_deg=float(g.lon[ix]),lat_bounds_deg=[float(g.lat[north]),float(g.lat[south])],
        control_wet_rows=channel.tolist(),meaning='fixed meridional section between control island north tip and mainland; not minimum navigable width',
        cell_height_km=float(6371*np.pi/len(g.lat)),cases={})
    for k in land:
        wet=~land[k][channel,ix]
        section['cases'][k]=dict(all_section_wet=bool(wet.all()),
            wet_cells=int(wet.sum()),total_cells=len(channel),
            summed_wet_cell_height_km=float(wet.sum()*6371*np.pi/len(g.lat)),
            I4_connected_to_main=summ['cases'][k]['I4_probe_connected_to_main'])
    summ['fixed_I4_section']=section
    summ['causal_evidence']=dict(W_birth=runs['control']['watched_births'],
        E_initial_uid=66385,E_initial_status='present in inherited initial300Ma native state; age302Ma at0Ma; not a demonstrated trench',
        limits=['birth nearest-pair recomputed at assigned parcel center, not original fixed gap integration point',
            'W default ocean is commit gap infill, not by itself evidence of ocean spreading or rollback',
            'ports use Engine.covering approximate cells, not exact shared physical boundaries',
            'partition does not establish weak rheology; broad I4 inherited design, fine coast unresolved'])
    fig,axs=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    d=native['control'];lo,la=xyz_to_lonlat(d['positions']);pick=d['alive']&(lo>-14)&(lo<15)&(la>32)&(la<52)
    with (ROOT/'.cache/u44_pockets/u44_partition_control/final_state.pkl').open('rb') as f:eng=pickle.load(f)
    ports,cov=covered_boundary_ports(eng)
    closed,opened,part=thin_partition(d['T'],d['edges'],d['alive'],d['ctype'],d['plate'],d['origin'],
        load_mesh(7).degree[d['node']],boundary_ports=ports)
    cats=np.where(closed,2,np.where(opened,1,0))
    from matplotlib.colors import ListedColormap
    im=axs[0].scatter(lo[pick],la[pick],c=cats[pick],cmap=ListedColormap(['#d6cdbb','#276c99','#d65e41']),
        vmin=-.5,vmax=2.5,s=18,marker='h')
    axs[0].set(xlim=(-14,15),ylim=(32,52),xlabel='经度 / °',ylabel='纬度 / °',title='控制末态拓扑：陆缘与封闭薄壳')
    axs[0].set_aspect(1/np.cos(np.deg2rad(42)));axs[0].grid(alpha=.2)
    cb=fig.colorbar(im,ax=axs[0],orientation='horizontal',fraction=.04,pad=.08,ticks=[0,1,2])
    cb.ax.set_xticklabels(['其他','开放薄壳','封闭薄壳'],fontsize=8)
    for k,label_,color in [('control','控制','#333333'),('uniform','统一交换','#bc3c2a'),('pockets','封闭分区','#246eaf')]:
        sel=(g.lat>36)&(g.lat<45)
        axs[1].plot(g.lat[sel],z[k][sel,ix],label=label_,color=color)
    axs[1].axhline(0,c='gray',ls=':');axs[1].axvspan(g.lat[north],g.lat[south],alpha=.12,color='blue')
    axs[1].set(xlabel='纬度 / °N',ylabel='构造尺度高程 / m',title=f'固定{g.lon[ix]:.3f}°E断面；蓝带为原海道')
    axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
    axs[2].axis('off')
    birth=runs['control']['watched_births']
    lines=['同位置海道与实际UID事件','断面湿格高度和≠真实最窄宽度']
    for k,label_ in [('control','控制'),('uniform','统一'),('pockets','分区')]:
        q=section['cases'][k];lines.append(f"{label_}：湿格{q['wet_cells']}/{q['total_cells']}，I4合陆={q['I4_connected_to_main']}")
    lines+=['','W：UID470099在实际commit中首次出现']
    for q in birth:
        if q['uid']==470099:
            lines += [f"提交时间{q['t_ma']:g}Ma；归属{q['plate']}",
                f"相邻板块{q['nearest_pair']}；壳型{q['nearest_ctype']}",
                f"局部相对速度点积{q['opening_dot_rad_myr']:.2e}",
                '邻对在分配后单元中心复查，非原gap采样点']
    lines+=['E：UID66385继承初始老洋壳','年轻补洞≠已证明扩张/后撤','分区是假设，细岸成因仍需解释']
    axs[2].text(0,.98,'\n'.join(lines),va='top',fontsize=9,linespacing=1.8)
    summ['control_final_partition']=dict(partition=part,coverage=cov)
    save(fig,'09_channel_partition.png','I4海道固定断面、原生分区与W/E实际事件证据；不是航道宽度或真实流变证明')
