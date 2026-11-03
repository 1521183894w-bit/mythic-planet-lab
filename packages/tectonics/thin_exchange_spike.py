"""U43：WT地体旧薄陆壳门槛的有限交换假设；非经标定流变。"""
from __future__ import annotations
import numpy as np
from mythic.tectonics.crust import CONT
from mythic.tectonics.relaxation_spike import relax_thickness

def thin_exchange(T, area, edges, alive, ctype, plate, origin, dt_myr, h_km,
                  kappa_km2_myr, onset_km=20., scale_km=20., target_plate=302):
    """沿WT存活陆壳原生边叠加弱背景交换；不设32.5km断流边界，不改壳型。"""
    if not np.isfinite(kappa_km2_myr) or kappa_km2_myr < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if not np.isfinite([onset_km, scale_km]).all() or scale_km <= 0:
        raise ValueError("invalid response parameters")
    a, b = np.asarray(edges, dtype=np.int64).T
    local = alive & (ctype == CONT) & (plate == target_plate)
    eligible = (local[a] & local[b]
                & ((origin[a] == target_plate) | (origin[b] == target_plate))
               )
    e = np.asarray(edges)[eligible]
    if kappa_km2_myr == 0 or not len(e):
        return T.copy(), np.zeros_like(T), dict(edges=int(len(e)), balance_km3=0.,
            max_local_residual_km3=0., max_delta_T_km=0., net_received_km3=0.)
    out, dv, info = relax_thickness(T, area, e, np.ones(len(e)), dt_myr, h_km,
                                   kappa_km2_myr, onset_km=onset_km, scale_km=scale_km)
    info.update(edges=int(len(e)), max_delta_T_km=float(np.max(np.abs(out-T))),
                net_received_km3=float(dv[dv>0].sum()))
    return out, dv, info
