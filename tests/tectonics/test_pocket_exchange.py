"""U44结构分类的开放端口反例及有限交换；不把测试通过当世界通过。"""
import numpy as np
import pytest
from mythic.tectonics.pocket_exchange_spike import thin_partition,pocket_exchange

def partition(T,e,**kw):
    n=len(T);args=dict(alive=np.ones(n,bool),ctype=np.ones(n,int),
        plate=np.full(n,302),origin=np.full(n,302),
        full_degree=np.bincount(np.array(e).ravel(),minlength=n))
    args.update(kw)
    return thin_partition(np.array(T,float),np.array(e),**args)

def test_closed_patch_and_ocean_connected_chain_are_separate():
    e=[[0,1],[1,2],[2,3],[3,4]];T=[25,38,31,30,7]
    closed,opened,_=partition(T,e,ctype=np.array([1,1,1,1,0]))
    assert np.array_equal(closed,[1,0,0,0,0])
    assert np.array_equal(opened,[0,0,1,1,0])

def test_missing_neighbor_is_open_port_even_without_ocean():
    closed,opened,_=partition([25,38],[[0,1]],full_degree=np.array([2,1]))
    assert not closed.any() and opened[0]

def test_dead_ocean_neighbor_is_still_missing_port():
    closed,opened,_=partition([25,38,7],[[0,1],[0,2]],
                              alive=np.array([1,1,0],bool),ctype=np.array([1,1,0]))
    assert not closed.any() and opened[0]

def test_partition_is_invariant_under_node_permutation():
    T=np.array([25,38,31,30,7.]);e=np.array([[0,1],[1,2],[2,3],[3,4]])
    ct=np.array([1,1,1,1,0]);c,o,_=partition(T,e,ctype=ct)
    perm=np.array([4,1,0,3,2]);inv=np.argsort(perm)
    c2,o2,_=partition(T[perm],inv[e],ctype=ct[perm])
    assert np.array_equal(c,c2[inv]) and np.array_equal(o,o2[inv])

def test_exchange_preserves_open_margin_and_volume_and_extrema():
    T=np.array([25,38,31,30,7.]);e=np.array([[0,1],[1,2],[2,3],[3,4]])
    ct=np.array([1,1,1,1,0]);c,o,_=partition(T,e,ctype=ct)
    args=(T,np.arange(1,6),e,np.ones(5,bool),ct,np.full(5,302),np.full(5,302),c,o,1.,60.)
    new,dv,row=pocket_exchange(*args)
    assert 25<new[0]<new[1]<38
    assert np.array_equal(new[2:],T[2:])
    assert abs(np.dot(new-T,np.arange(1,6)))<1e-10
    assert abs(dv.sum())<1e-10
    assert row['max_local_residual_km3']<1e-10
    zero,_,_=pocket_exchange(*args,kappa_km2_myr=0)
    assert np.array_equal(zero,T)

def test_remembered_patch_can_evolve_above_old_threshold():
    new,_,_=pocket_exchange(np.array([32.6,38.]),np.ones(2),np.array([[0,1]]),
        np.ones(2,bool),np.ones(2,int),np.full(2,302),np.full(2,302),
        np.array([1,0],bool),np.zeros(2,bool),1.,60.)
    assert new[0]>32.6

@pytest.mark.parametrize('kappa',[-1.,float('nan'),float('inf')])
def test_invalid_coefficient(kappa):
    with pytest.raises(ValueError):
        pocket_exchange(np.array([25.,38.]),np.ones(2),np.array([[0,1]]),
            np.ones(2,bool),np.ones(2,int),np.full(2,302),np.full(2,302),
            np.array([1,0],bool),np.zeros(2,bool),1.,60.,kappa)

def test_missing_native_site_covered_by_foreign_continent_is_closed():
    from types import SimpleNamespace
    from mythic.tectonics.pocket_exchange_spike import covered_boundary_ports
    class FakeEngine:
        def covering(self,q,pts):return np.full(len(pts),1,dtype=int)
    eng=FakeEngine()
    from mythic.tectonics.crust import Crust
    eng.c=Crust(2)
    eng.c.T[:]=[25.,38.];eng.c.ctype[:]=1
    eng.c.plate[:]=[302,103];eng.c.node[:]=[0,2]
    eng.alive=np.ones(2,bool)
    eng.m=SimpleNamespace(nbrs=np.array([1,0,1]),indptr=np.array([0,1,2,3]),
        xyz=np.eye(3))
    eng.owner={302:np.array([0,-1,-1])}
    eng.R={302:np.eye(3)}
    eng.plates=[302,103]
    ports,row=covered_boundary_ports(eng)
    assert not ports.any() and row['foreign_continent_covered']==1
    eng.c.ctype[1]=0
    ports,row=covered_boundary_ports(eng)
    assert ports[0] and row['open_links']==1
