"""U43新增交换的守恒、边界隔离和弱壳启动条件。"""
import numpy as np
import pytest
from mythic.tectonics.thin_exchange_spike import thin_exchange
from mythic.tectonics.relaxation_spike import relax_thickness

def run(T,edges,**kwargs):
    T=np.array(T,float);n=len(T)
    args=dict(area=np.arange(n,dtype=float)+1,edges=np.array(edges).reshape(-1,2),
        alive=np.ones(n,bool),ctype=np.ones(n,int),plate=np.full(n,302),
        origin=np.full(n,302),dt_myr=1.,h_km=60.,kappa_km2_myr=500.)
    args.update(kwargs)
    return thin_exchange(T,**args)

def test_actual_below40_pair_can_exchange_without_source():
    T=np.array([25.,38.])
    old,_,_=relax_thickness(T,np.array([1.,2.]),np.array([[0,1]]),
        np.ones(1),1.,60.,500.)
    assert np.array_equal(old,T)
    new,dv,info=run(T,[[0,1]])
    assert 25<new[0]<new[1]<38
    assert abs(np.dot(new-T,[1,2]))<1e-11
    assert abs(dv.sum())<1e-11
    assert info['max_local_residual_km3']<1e-11

def test_cross_plate_ocean_and_dead_edges_never_receive():
    new,dv,_=run([38.,25.,24.,23.,22.],[[0,1],[0,2],[0,3],[0,4]],
        plate=np.array([302,302,103,302,302]),
        ctype=np.array([1,1,1,0,1]),alive=np.array([1,1,1,1,0],bool))
    assert np.array_equal(new[2:],[24.,23.,22.])
    assert np.array_equal(dv[2:],[0.,0.,0.])

def test_zero_coupling_exactly_preserves_control():
    old=np.array([24.,38.]);new,dv,_=run(old,[[0,1]],kappa_km2_myr=0.)
    assert np.array_equal(new,old)
    assert np.array_equal(dv,[0.,0.])

def test_no32_point5_discontinuity_in_background_pair():
    low=run([32.49,38.],[[0,1]])[0]
    high=run([32.51,38.],[[0,1]])[0]
    assert low[0]>32.49 and high[0]>32.51
    assert np.max(abs(low-high))<.021

@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf')])
def test_rejects_invalid_coefficients(bad):
    with pytest.raises(ValueError):run([25.,38.],[[0,1]],kappa_km2_myr=bad)
