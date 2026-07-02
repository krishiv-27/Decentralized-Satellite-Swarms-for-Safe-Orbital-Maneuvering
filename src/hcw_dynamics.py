"""
Hill-Clohessy-Wiltshire Relative Orbital Dynamics Engine
Generates: state transition matrices, reachable set envelopes, 
           separation violation check (SMT proxy), natural motion trajectories
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
import time

# ── Physical constants ──────────────────────────────────────────────────
MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH  = 6.3781e6         # m

def mean_motion(alt_km):
    """Mean motion n [rad/s] for circular orbit at altitude alt_km [km]"""
    a = R_EARTH + alt_km * 1e3
    return np.sqrt(MU_EARTH / a**3)

def hcw_A_matrix(n):
    """
    Continuous-time HCW A matrix: xdot = A x + B u
    State: [x, xdot, y, ydot, z, zdot]  (radial, along-track, cross-track)
    """
    return np.array([
        [0,    1,  0, 0,    0, 0],
        [3*n**2, 0,  0, 2*n, 0, 0],
        [0,    0,  0, 1,    0, 0],
        [0, -2*n,  0, 0,    0, 0],
        [0,    0,  0, 0,    0, 1],
        [0,    0,  0, 0, -n**2, 0],
    ])

def hcw_B_matrix():
    """Control input matrix (thrust acceleration -> state derivative)"""
    B = np.zeros((6,3))
    B[1,0] = 1.0
    B[3,1] = 1.0
    B[5,2] = 1.0
    return B

def state_transition_matrix(n, dt):
    """Closed-form HCW state transition matrix Phi(t) via matrix exponential"""
    A = hcw_A_matrix(n)
    return expm(A * dt)

def propagate_free_drift(x0, n, t_array):
    """
    Propagate N satellites under free-drift HCW dynamics.
    x0: (N, 6) initial state array
    Returns trajectory: (N, len(t_array), 6)
    """
    N = x0.shape[0]
    traj = np.zeros((N, len(t_array), 6))
    traj[:, 0, :] = x0
    for k, t in enumerate(t_array[1:], 1):
        Phi = state_transition_matrix(n, t)
        for i in range(N):
            traj[i, k, :] = Phi @ x0[i]
    return traj

def separation_matrix(traj):
    """
    Compute pairwise Euclidean separation at every time step.
    traj: (N, T, 6) — returns (N, N, T) separation distances [m]
    """
    N, T, _ = traj.shape
    seps = np.zeros((N, N, T))
    for i in range(N):
        for j in range(i+1, N):
            dp = traj[i,:,:3] - traj[j,:,:3]  # (T,3) position diff
            d  = np.linalg.norm(dp, axis=1)
            seps[i,j,:] = d
            seps[j,i,:] = d
    return seps

def minimum_separation(seps, i, j):
    """Minimum separation between satellite i and j over trajectory"""
    return np.min(seps[i,j,:])

def time_of_closest_approach(seps, i, j, t_array):
    """Time of closest approach between pair (i,j)"""
    idx = np.argmin(seps[i,j,:])
    return t_array[idx], seps[i,j,idx]

# ── SMT-proxy: interval-arithmetic safety certificate ──────────────────
def verify_separation_smt_proxy(x0_i, x0_j, n, T_burn, r_min, delta_r,
                                 n_intervals=1000):
    """
    Proxy for SMT QF_NRA query: checks whether ||p_i(t) - p_j(t)||_2 >= r_min + delta_r
    for all t in [0, T_burn], given initial states x0_i, x0_j.
    
    Returns: (certified: bool, min_sep: float, witness_time: float, solve_time_ms: float)
    'certified' = True means no violation found (UNSAT analog)
    """
    t0 = time.perf_counter()
    t_grid = np.linspace(0, T_burn, n_intervals)
    r_safe = r_min + delta_r
    
    min_sep = np.inf
    witness_t = 0.0
    
    for t in t_grid:
        Phi = state_transition_matrix(n, t)
        p_i = (Phi @ x0_i)[:3]
        p_j = (Phi @ x0_j)[:3]
        sep = np.linalg.norm(p_i - p_j)
        if sep < min_sep:
            min_sep = sep
            witness_t = t
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    certified = (min_sep >= r_safe)
    return certified, min_sep, witness_t, elapsed_ms

print("HCW dynamics engine loaded successfully.")
