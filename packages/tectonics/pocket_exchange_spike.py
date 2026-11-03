"""U44：原生拓扑区分封闭薄壳与开放陆缘的输运假设；不是流变/湖泊求解。"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from mythic.tectonics.relaxation_spike import relax_thickness

def thin_partition(T, edges, alive, ctype, plate, origin, full_degree, target_plate=302,
                   thin_threshold_km=32.5, boundary_ports=None):
    """沿薄壳原生边追踪开放端口；洋壳邻接或缺失原生邻点都视为开放。
    
    32.5 km沿用旧模型薄壳分界，只用于候选结构分类，不是已标定强度。
    不按经纬度、点名、面积或末态海陆筛选。死亡边不作有效邻接。
    """
    T=np.asarray(T,float);n=len(T)
    e=np.asarray(edges,np.int64).reshape(-1,2)
    if (not np.isfinite(T).all() or not np.isfinite(thin_threshold_km)
        or any(np.asarray(v).shape!=(n,) for v in [alive,ctype,plate,origin,full_degree])
        or np.any(e<0) or np.any(e>=n) or np.any(np.asarray(full_degree)<0)):
        raise ValueError('invalid partition input')
    a,b=e.T
    live=np.asarray(alive,bool)
    local=live&(np.asarray(ctype)==1)&(np.asarray(plate)==target_plate)
    thin=local&(T<thin_threshold_km)
    ids=np.flatnonzero(thin)
    closed=np.zeros(n,bool);opened=closed.copy()
    if not len(ids):
        return closed,opened,dict(thin_cells=0,closed_cells=0,open_cells=0,closed_components=0)
    degree=np.bincount(e[live[a]&live[b]].ravel(),minlength=n)
    ports=thin&(degree<np.asarray(full_degree))
    if boundary_ports is not None:
        bp=np.asarray(boundary_ports,bool)
        if bp.shape!=(n,):raise ValueError("invalid boundary ports")
        ports=thin&bp
    # 存活洋壳、其他板块端点或其他壳型都不能作为封闭陆壳围壁。
    ports[a[thin[a]&live[b]&~local[b]]]=True
    ports[b[thin[b]&live[a]&~local[a]]]=True
    ren=np.full(n,-1,np.int64);ren[ids]=np.arange(len(ids))
    te=e[thin[a]&thin[b]]
    graph=coo_matrix((np.ones(len(te)*2),(np.r_[ren[te[:,0]],ren[te[:,1]]],
                     np.r_[ren[te[:,1]],ren[te[:,0]]])),shape=(len(ids),len(ids))).tocsr()
    count,lab=connected_components(graph,directed=False)
    open_labels=np.unique(lab[ports[ids]])
    is_open=np.isin(lab,open_labels)
    opened[ids[is_open]]=True;closed[ids[~is_open]]=True
    # 只允许原WT继承陆壳成为弱单元；开放域分类仍包括同板块其他来源。
    closed &= np.asarray(origin)==target_plate
    return closed,opened,dict(thin_cells=len(ids),closed_cells=int(closed.sum()),
        open_cells=int(opened.sum()),closed_components=int(count-len(open_labels)))

def pocket_exchange(T,area,edges,alive,ctype,plate,origin,weak,protected,dt_myr,h_km,
                    kappa_km2_myr=100.,target_plate=302):
    """只在已识别弱斑块的邻接边上作成对交换，开放薄壳端点不接收/输出。
    
    弱斑块通过UID持久继承，升过分类厚度仍可渐进交换；不把32.5km当高程目标。
    原生边、冻结系数后向欧拉、20km启动/响应尺度均复用U43。
    """
    if not np.isfinite(kappa_km2_myr) or kappa_km2_myr<0:
        raise ValueError('invalid diffusivity')
    T=np.asarray(T,float);e=np.asarray(edges,np.int64).reshape(-1,2);a,b=e.T
    local=np.asarray(alive,bool)&(np.asarray(ctype)==1)&(np.asarray(plate)==target_plate)
    w=np.asarray(weak,bool);p=np.asarray(protected,bool)
    if w.shape!=T.shape or p.shape!=T.shape:raise ValueError('invalid classification shape')
    ok=local[a]&local[b]&~p[a]&~p[b]&(w[a]|w[b])
    ok &= (np.asarray(origin)[a]==target_plate)|(np.asarray(origin)[b]==target_plate)
    out,dv,row=relax_thickness(T,area,e[ok],np.ones(int(ok.sum())),dt_myr,h_km,
                             kappa_km2_myr,onset_km=20.,scale_km=20.)
    row.update(edges=int(ok.sum()),max_delta_T_km=float(abs(out-T).max()),
               net_received_km3=float(dv[dv>0].sum()))
    return out,dv,row

def covered_boundary_ports(engine):
    """缺失的原生邻节点仅在被别块存活陆壳实际覆盖时视为封闭围壁。
    
    复用Engine.covering的最近网格覆盖语义；不是精确共边或力学接触。
    未覆盖、被洋壳覆盖均开放；不以终态海陆或地理点名反推。
    """
    c=engine.c;m=engine.m
    thin=engine.alive&(c.ctype==1)&(c.plate==302)&(c.T<32.5)
    ports=np.zeros(len(c),bool);sources=[];nodes=[]
    for i in np.flatnonzero(thin):
        ns=m.nbrs[m.indptr[c.node[i]]:m.indptr[c.node[i]+1]]
        own=engine.owner[302][ns]
        has=own>=0
        has[has] &= engine.alive[own[has]]
        missing=ns[~has]
        sources.extend([i]*len(missing));nodes.extend(missing.tolist())
    if not nodes:return ports,dict(missing_links=0,foreign_continent_covered=0,open_links=0)
    pts=m.xyz[np.array(nodes)]@engine.R[302].T
    covered=np.zeros(len(pts),bool)
    for q in engine.plates:
        if q==302:continue
        host=engine.covering(q,pts);valid=host>=0
        valid[valid] &= c.ctype[host[valid]]==1
        covered |= valid
    ports[np.asarray(sources)[~covered]]=True
    return ports,dict(missing_links=len(nodes),foreign_continent_covered=int(covered.sum()),
                      open_links=int((~covered).sum()))
