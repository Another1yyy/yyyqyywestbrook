# -*- coding: utf-8 -*-
"""
Problem 5 Verifier (Point Target, Triple-Missile Simultaneous Occlusion)
- Physics identical to the Q4 verifier:
  * Cloud: radius 10 m, sinks 3 m/s, lifetime 20 s
  * Missiles: 300 m/s, straight to origin from given M0
  * Target treated as a POINT T=(0,200,0)
  * Occlusion iff dist( cloud_center, segment[M(t) -> T] ) <= 10 m
- New in Q5:
  * Count time only when ALL THREE missiles (M1,M2,M3) are covered simultaneously
- Inputs:
  * missiles: list of dict with {'name','M0': np.array([x,y,z])}
  * uavs: 5 UAV dicts, each with {'name','F0','v','theta_deg','t_drop','tau'}
    (t_drop/tau can be scalar or list; up to 3 bombs per UAV)
- Outputs:
  * triple_total_time, triple_intervals, per_missile_union_intervals, per_bomb_infos
Author: Westbrook专用（从Q4校验器泛化）
"""

import math
import numpy as np

# =========================
# Global constants (from problem statement)
# =========================
g = 9.8
R_cloud = 10.0        # cloud effective radius [m]
smoke_duration = 20.0 # cloud lifetime [s]
sink_v = 3.0          # cloud sinks down [m/s]
vm = 300.0            # missile speed [m/s]
EVAL_DT = 0.02        # time step (smaller => finer/slow; bigger => faster/rough)

# Target point (treat the true target as a point)
T_POINT = np.array([0.0, 200.0, 0.0], dtype=float)

# =========================
# Default initial positions from the problem (overridable)
# =========================
M1_M0 = np.array([20000.0,     0.0, 2000.0])
M2_M0 = np.array([19000.0,   600.0, 2100.0])
M3_M0 = np.array([18000.0,  -600.0, 1900.0])

FY1_F0 = np.array([17800.0,    0.0, 1800.0])
FY2_F0 = np.array([12000.0,  1400.0, 1400.0])
FY3_F0 = np.array([ 6000.0, -3000.0,  700.0])
FY4_F0 = np.array([11000.0,  2000.0, 1800.0])
FY5_F0 = np.array([13000.0, -2000.0, 1300.0])

DEFAULT_F0 = {"FY1": FY1_F0, "FY2": FY2_F0, "FY3": FY3_F0, "FY4": FY4_F0, "FY5": FY5_F0}

# =========================
# Interval utilities
# =========================
def merge_intervals(intervals):
    """Merge overlapping/adjacent intervals (numerical epsilon included)."""
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb + 1e-9:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def intersect_two(A, B):
    """Intersect two interval sets A and B (each already merged)."""
    i = j = 0
    out = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i]; b1, b2 = B[j]
        s, e = max(a1, b1), min(a2, b2)
        if s <= e - 1e-12:
            out.append((s, e))
        if a2 < b2: i += 1
        else: j += 1
    return out

def intersect_many(sets):
    """Intersect a list of interval sets (merged)."""
    if not sets: return []
    ans = sets[0]
    for s in sets[1:]:
        ans = intersect_two(ans, s)
        if not ans: break
    return ans

def total_length(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)

# =========================
# Kinematics
# =========================
def missile_dir_u_and_hit_time(M0):
    """Unit direction (toward origin) and hit time (to reach origin)."""
    d = np.linalg.norm(M0)
    u = -M0 / d
    return u, d / vm

def missile_pos_vec(t, M0, u_m):
    """Missile position at absolute time t (vectorized)."""
    t = np.asarray(t, dtype=float)
    return M0[None, :] + (vm * t[:, None]) * u_m[None, :]

def explosion_point(F0, v, theta_rad, t_drop, tau):
    """
    From (F0, v, θ, t_drop, τ) compute:
    - drop point R
    - explosion point E
    - explosion time t_e = t_drop + tau
    θ: heading in xy-plane (x-axis 0 rad, CCW positive)
    """
    h = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
    R = F0 + v * t_drop * h
    E = R + v * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)
    return R, E, (t_drop + tau)

# =========================
# Occlusion signal for ONE bomb vs ONE missile
# =========================
def occlusion_intervals_point_target_for_missile(E, t_e, M0, dt=EVAL_DT):
    """
    Given a single cloud (explosion at E at time t_e), compute its occlusion intervals
    w.r.t one missile started at M0 toward origin, point target T_POINT.
    """
    u_m, T_HIT = missile_dir_u_and_hit_time(M0)
    # cloud active window clipped by cloud lifetime and missile existence before hitting origin
    span = min(smoke_duration, max(0.0, T_HIT - t_e))
    if span <= 0.0: return []

    # time grids
    T = int(math.floor(span / dt)) + 1
    tgrid = np.linspace(0.0, span, T)
    tabs = t_e + tgrid

    # positions
    M = missile_pos_vec(tabs, M0, u_m)  # (T,3)
    # cloud center (downwards after explosion)
    C = E[None, :] + np.c_[np.zeros_like(tgrid), np.zeros_like(tgrid), -sink_v * tgrid]

    # distance from C to segment [M(t) -> T_POINT]
    S = T_POINT[None, :] - M           # (T,3)
    L2 = np.einsum('ij,ij->i', S, S)
    L2 = np.where(L2 > 1e-12, L2, 1e-12)
    CM = C - M
    s = np.clip(np.einsum('ij,ij->i', CM, S) / L2, 0.0, 1.0)
    Q = M + s[:, None] * S
    d = np.linalg.norm(C - Q, axis=1)

    inside = d <= R_cloud
    if not np.any(inside): return []

    intervals = []
    i = 0
    while i < T - 1:
        if inside[i]:
            j = i + 1
            while j < T and inside[j]:
                j += 1
            # entry time (linear interpolation)
            t_in = tgrid[i]
            if i > 0 and not inside[i - 1]:
                f1, f2 = d[i - 1] - R_cloud, d[i] - R_cloud
                t_in = tgrid[i - 1] + (tgrid[i] - tgrid[i - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            # exit time
            t_out = tgrid[j - 1]
            if j < T and not inside[j]:
                f1, f2 = d[j - 1] - R_cloud, d[j] - R_cloud
                t_out = tgrid[j - 1] + (tgrid[j] - tgrid[j - 1]) * (abs(f1) / (abs(f1) + abs(f2)))
            intervals.append((t_e + t_in, t_e + t_out))
            i = j
        else:
            i += 1
    return merge_intervals(intervals)

# =========================
# Main verification
# =========================
def verify_triple_occlusion_time_point_target(
    missiles,
    uav_list,
    dt=EVAL_DT,
    print_details=True
):
    """
    missiles: list of 3 dicts:
        {'name': 'M1', 'M0': np.array([x,y,z])}, etc.
    uav_list: list of up to 5 dicts:
        {
            "name": "FY1",
            "F0": np.array([x0,y0,z0])  # optional; default from problem statement
            "v":  110.0,
            "theta_deg": 30.0,
            "t_drop": [10.0, 18.0],     # scalar or list
            "tau":    [2.0, 2.5],       # scalar or list (broadcastable)
        }
    Returns:
        triple_total_time: float
        triple_intervals: list[(t_start, t_end)]  # intersection(M1_union, M2_union, M3_union)
        per_missile_union_intervals: dict{name: list[(...)]}
        per_bomb_infos: list of per-cloud info tuples
            (uav_name, bomb_idx, missile_name, cloud_Ez, t_e, own_dur, own_first_iv or None)
    """
    # Normalize missiles
    missiles_norm = []
    for m in missiles:
        name = m.get("name", "M?")
        M0 = np.array(m["M0"], dtype=float)
        missiles_norm.append((name, M0))

    # Build all clouds (explosions) from UAV inputs
    clouds = []  # list of dict {name, E, t_e}
    per_bomb_infos = []

    for uav in uav_list:
        name = uav.get("name", "FY?")
        F0 = np.array(uav.get("F0", DEFAULT_F0.get(name, FY1_F0)), dtype=float)
        v = float(uav["v"])
        th = math.radians(float(uav["theta_deg"]))
        t_drop = uav["t_drop"]
        tau = uav["tau"]
        # broadcast to list
        if np.isscalar(t_drop): t_drop = [float(t_drop)]
        if np.isscalar(tau):    tau    = [float(tau)] * len(t_drop)
        assert len(t_drop) == len(tau), f"{name}: t_drop and tau length mismatch"

        for k, (td, ta) in enumerate(zip(t_drop, tau), start=1):
            R, E, t_e = explosion_point(F0, v, th, float(td), float(ta))
            if E[2] <= 0.0:
                if print_details:
                    print(f"[WARN] {name}#{k} explosion Ez={E[2]:.3f} <= 0 -> skip")
                continue
            clouds.append(dict(uav=name, idx=k, E=E, t_e=t_e))

    # For each missile, union the occlusion intervals from all clouds
    per_missile_union = {}
    for m_name, M0 in missiles_norm:
        acc = []
        for c in clouds:
            ivs = occlusion_intervals_point_target_for_missile(c["E"], c["t_e"], M0, dt=dt)
            if ivs:
                acc.extend(ivs)
                # record own-bomb stats for debugging/inspection
                own_dur = total_length(ivs)
                per_bomb_infos.append((
                    c["uav"], c["idx"], m_name,
                    c["E"][2], c["t_e"], own_dur,
                    ivs[0] if ivs else None
                ))
        per_missile_union[m_name] = merge_intervals(acc)

    # Intersect the three missiles' union intervals
    all_sets = [per_missile_union[m_name] for m_name, _ in missiles_norm]
    triple_intervals = intersect_many(all_sets)
    triple_total_time = total_length(triple_intervals)

    if print_details:
        print("\n=== Q5 Verification (Point Target; Triple-Missile Simultaneous) ===")
        for m_name, _ in missiles_norm:
            ivs = per_missile_union[m_name]
            print(f"- {m_name} union cover: {total_length(ivs):.6f} s, {len(ivs)} intervals")
            for i, (a, b) in enumerate(ivs, 1):
                print(f"  · {m_name}[{i}] = [{a:.6f}, {b:.6f}]  dur={b-a:.6f}")
        print(f"\n>>> Triple (M1∩M2∩M3) TOTAL = {triple_total_time:.6f} s, {len(triple_intervals)} intervals")
        for i, (a, b) in enumerate(triple_intervals, 1):
            print(f"  · Triple[{i}] = [{a:.6f}, {b:.6f}]  dur={b-a:.6f}")

    return triple_total_time, triple_intervals, per_missile_union, per_bomb_infos


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Missiles (default from problem; can be overridden)
    MISSILES = [
        dict(name="M1", M0=M1_M0),
        dict(name="M2", M0=M2_M0),
        dict(name="M3", M0=M3_M0),
    ]

    # Example UAVs (fill with your search/strategy; each UAV up to 3 bombs)
    UAVS = [
        dict(name="FY1", v=140.0, theta_deg=180, t_drop=[0.000, 1.5891357, 4.49948], tau=[0, 4.504, 5.73]),
        dict(name="FY2", v=98.227153, theta_deg=282.483654,
             t_drop=[7.198853514385463, 8.198854514385461, 49.992508461305576],
             tau=[2.9630817130408045, 2.192383209103564, 5.03745550276874]),
        dict(name="FY3", v=140.0, theta_deg=82.250, t_drop=[18.714, 19.714, 22.423], tau=[1.982, 2.490, 11.951]),
        dict(name="FY4", v=125.811, theta_deg=-90.036, t_drop=[1.899, 3.608, 41.055], tau=[10.855, 11.893, 4.491]),
        dict(name="FY5", v=111.065118, theta_deg=122.540507, t_drop=[0.0, 14.533026756493195, 55.59243510604091],
             tau=[7.333341859137445, 2.6121651315338434, 15.0]),
    ]

    triple_time, triple_ivs, per_m_union, per_bombs = verify_triple_occlusion_time_point_target(
        missiles=MISSILES, uav_list=UAVS, dt=EVAL_DT, print_details=True
    )
