# -*- coding: utf-8 -*-
"""
Problem 3 (Single UAV FY1, 3 bombs) — Differential Evolution
- Decision vars: 3 explosion points [(x,y,z,t)*3]
- Inverse model recovers (v_uav, theta, t_drop, tau) for each bomb
- Objective: maximize UNION cover time for missile M1 (20s smoke life, sink 3 m/s)
- Enforce: per-UAV min drop-gap >= 1s; speed & delay bounds; E_z < F0_z; t_drop >= 0
- Output: Excel (openpyxl) if available, otherwise CSV fallback

Dependencies:
    numpy, pandas, scipy (scipy.optimize.differential_evolution)
"""

import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time
import os

# ==============================
# 1) Constants & global params
# ==============================
g = 9.8
R_eff = 10.0            # cloud effective radius
smoke_duration = 20.0   # 20 s lifetime
sink_speed = 3.0        # m/s downward
dt_eval = 0.1           # time sampling step for masks (can raise to 0.12 for more speed)

# Scene & kinematics
O = np.array([0.0, 0.0, 0.0])
T_true = np.array([0.0, 200.0, 0.0])   # true target point (与第4问脚本一致)
vm = 300.0                              # missile speed

# Missiles (we only use M1 here, but keep structure for reuse)
MISSILES_INFO = {
    'M1': {'M0': np.array([20000., 0., 2000.])},
    'M2': {'M0': np.array([19000., 600., 2100.])},
    'M3': {'M0': np.array([18000., -600., 1900.])}
}

# Drones (only FY1 is used for Problem 3)
DRONES_INFO = {
    'FY1': {'F0': np.array([17800., 0., 1800.])},
}

# Feasible bounds
V_UAV_BOUNDS = (70.0, 140.0)  # m/s
TAU_BOUNDS   = (0.2, 12.0)    # s

# ==============================
# 2) Physics helpers
# ==============================
def pre_calculate_missile_trajectories(missiles):
    """Precompute straight-line missile trajectories toward the origin."""
    trajectories = {}
    for name, data in missiles.items():
        um = (O - data['M0']) / np.linalg.norm(O - data['M0'])
        T_hit = np.linalg.norm(O - data['M0']) / vm
        times = np.arange(0, T_hit + dt_eval, dt_eval)
        positions = data['M0'] + vm * times[:, None] * um
        trajectories[name] = {'times': times, 'positions': positions, 'T_hit': T_hit}
    return trajectories

def inverse_model(explosion_point, F0):
    """
    Given an explosion point (x,y,z,t_exp), compute a feasible UAV plan:
    (v_uav, theta, t_drop, tau). Returns {'is_reachable': False} if infeasible.
    """
    x_exp, y_exp, z_exp, t_exp = explosion_point
    x_f0, y_f0, z_f0 = F0

    # Geometric/temporal sanity
    if t_exp <= 0 or z_exp > z_f0 or (z_f0 - z_exp) < 0:
        return {'is_reachable': False}

    # Vertical delay from z_f0 to z_exp under gravity (downward component)
    tau = np.sqrt(2.0 * (z_f0 - z_exp) / g)
    if not (TAU_BOUNDS[0] <= tau <= TAU_BOUNDS[1]):
        return {'is_reachable': False}

    t_drop = t_exp - tau
    if t_drop < 0:
        return {'is_reachable': False}

    # Horizontal kinematics: constant-speed along heading
    dx, dy = x_exp - x_f0, y_exp - y_f0
    horiz_dist = np.hypot(dx, dy)
    v = horiz_dist / t_exp if t_exp > 1e-6 else np.inf
    if not (V_UAV_BOUNDS[0] <= v <= V_UAV_BOUNDS[1]):
        return {'is_reachable': False}

    theta = np.arctan2(dy, dx)
    return {'is_reachable': True,
            'strategy': {'v_uav': v, 'theta': theta, 't_drop': t_drop, 'tau': tau}}

def get_shielding_mask(explosion_point, missile_traj):
    """
    Build a boolean mask over trajectory times where the cloud (radius R_eff)
    blocks the line segment from missile position to T_true.
    """
    full_mask = np.zeros_like(missile_traj['times'], dtype=bool)
    x_exp, y_exp, z_exp, t_exp = explosion_point

    # smoke active window on missile clock
    t0, t1 = t_exp, min(t_exp + smoke_duration, missile_traj['T_hit'])
    times = missile_traj['times']
    i0, i1 = np.searchsorted(times, [t0, t1])
    if i0 >= i1:
        return full_mask

    valid_t = times[i0:i1]
    Mt = missile_traj['positions'][i0:i1]            # (K,3)
    # cloud center: falls along -z after explosion
    Ct = np.array([x_exp, y_exp, z_exp])[None, :] + np.c_[np.zeros_like(valid_t),
                                                           np.zeros_like(valid_t),
                                                           -sink_speed * (valid_t - t_exp)]

    # distances from Ct to segment [Mt, T_true]
    seg_vec = (T_true - Mt)                           # (K,3)
    seg_len2 = np.sum(seg_vec**2, axis=1)
    seg_len2 = np.where(seg_len2 > 1e-9, seg_len2, 1e-9)
    CtMt = Ct - Mt
    tproj = np.clip(np.sum(CtMt * seg_vec, axis=1) / seg_len2, 0.0, 1.0)
    closest = Mt + tproj[:, None] * seg_vec
    d = np.linalg.norm(Ct - closest, axis=1)
    full_mask[i0:i1] = (d <= R_eff)
    return full_mask

# ==============================
# 3) Objective for Problem 3
# ==============================
def objective_problem3(X, missile_traj, drone_name='FY1', bombs_per_drone=3):
    """
    X shape: (bombs_per_drone*4,)  -> [x,y,z,t] for each bomb
    - Check reachability via inverse_model
    - Enforce per-UAV min gap 1s on t_drop
    - Return negative union cover time for M1 (to minimize)
    """
    F0 = DRONES_INFO[drone_name]['F0']
    points = X.reshape(bombs_per_drone, 4)

    # inverse model & gap check
    t_drops = []
    for point in points:
        inv = inverse_model(point, F0)
        if not inv['is_reachable']:
            return 1e9
        t_drops.append(inv['strategy']['t_drop'])
    t_drops.sort()
    for k in range(1, len(t_drops)):
        if t_drops[k] - t_drops[k-1] < 1.0:
            return 1e9

    # union mask over all bombs
    union_mask = np.zeros_like(missile_traj['times'], dtype=bool)
    for point in points:
        union_mask |= get_shielding_mask(point, missile_traj)

    total_union = np.sum(union_mask) * dt_eval
    return -total_union  # maximize by minimizing negative

# ==============================
# 4) Result saving (xlsx / csv)
# ==============================
def safe_save_results(results_list, filename_prefix="result_p3"):
    df = pd.DataFrame(results_list).round(6)
    # try excel via openpyxl
    try:
        out_xlsx = filename_prefix + ".xlsx"
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
        print(f"[OK] Results saved to {out_xlsx}")
    except Exception as e:
        out_csv = filename_prefix + ".csv"
        df.to_csv(out_csv, index=False)
        print(f"[WARN] Excel export failed ({e}); fallback to CSV -> {out_csv}")

# ==============================
# 5) Solve Problem 3
# ==============================
def solve_problem3_single_uav():
    print("\n" + "=" * 50)
    print("        SOLVING PROBLEM 3 (Single UAV FY1, 3 bombs)")
    print("=" * 50)

    # precompute missiles, use only M1 here
    all_traj = pre_calculate_missile_trajectories(MISSILES_INFO)
    mtraj = all_traj['M1']
    max_t_hit = mtraj['T_hit']

    # bounds for each explosion point (x,y,z,t)
    F0 = DRONES_INFO['FY1']['F0']
    # x in [0, F0_x + 2000], y in [-3000,3000], z in [0, F0_z-1], t in [1.0, T_hit]
    one_bomb_bounds = [(0, F0[0] + 2000), (-3000, 3000), (0, F0[2] - 1), (1.0, max_t_hit)]
    bounds = one_bomb_bounds * 3  # 3 bombs

    start = time.time()
    result = differential_evolution(
        func=objective_problem3,
        bounds=bounds,
        args=(mtraj, 'FY1', 3),
        maxiter=300, popsize=30,
        disp=True, workers=-1, polish=True
    )
    print(f"[INFO] Optimization finished in {time.time()-start:.2f} s.")

    if not result.success or result.fun >= 1e8:
        print("[FAILED] Optimization did not converge or infeasible.")
        return

    best_union = -result.fun
    print(f"\n[Problem 3 RESULT] Max UNION cover time (FY1, 3 bombs vs M1): {best_union:.4f} s")

    # decode strategies (and re-run inverse_model for reporting)
    points = result.x.reshape(3, 4)
    out_rows = []
    for i, point in enumerate(points, 1):
        inv = inverse_model(point, F0)
        st = inv['strategy']
        out_rows.append({
            '无人机': 'FY1',
            '烟幕弹序号': i,
            'UAV 航向 (deg)': np.degrees(st['theta']),
            'UAV 速度 (m/s)': st['v_uav'],
            '投放时刻 t_drop (s)': st['t_drop'],
            '起爆延时 tau (s)': st['tau'],
            '起爆时刻 t_exp (s)': st['t_drop'] + st['tau'],
            '起爆点 x': point[0], '起爆点 y': point[1], '起爆点 z': point[2],
        })

    # save
    safe_save_results(out_rows, filename_prefix="result_problem3")

if __name__ == "__main__":
    solve_problem3_single_uav()