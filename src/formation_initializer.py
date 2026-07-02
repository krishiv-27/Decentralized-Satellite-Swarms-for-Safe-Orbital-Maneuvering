"""
Formation geometry initializers for three experimental scenarios.
Returns (N, 6) HCW state arrays [x, xdot, y, ydot, z, zdot] in meters/m·s⁻¹.
"""
import numpy as np

def pco_formation(N: int, rho: float, n: float, rng=None):
    """
    Projected Circular Orbit (PCO) formation.
    N satellites equally spaced on an in-plane ellipse with semi-axis rho [m].
    Phase spacing: 2π/N. Bounded-drift condition: ydot_0 = -2n * x_0
    """
    if rng is None:
        rng = np.random.default_rng(0)
    states = np.zeros((N, 6))
    for k in range(N):
        theta = 2 * np.pi * k / N
        x0 = rho * np.cos(theta)
        y0 = -2 * rho * np.sin(theta)    # natural along-track offset
        xdot0 = -rho * n * np.sin(theta)
        ydot0 = -2 * n * x0              # bounded-drift condition
        zdot0 = 0.0
        z0    = 0.0
        states[k] = [x0, xdot0, y0, ydot0, z0, zdot0]
    return states

def string_of_pearls(N: int, spacing_m: float, rng=None):
    """
    Along-track string-of-pearls formation.
    N satellites separated by spacing_m in the y (along-track) direction.
    Pure along-track separation is orbitally stable under HCW (zero drift in y 
    requires xdot0=0, x0=0, ydot0=0).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    states = np.zeros((N, 6))
    for k in range(N):
        states[k, 2] = (k - N//2) * spacing_m   # y offset
    return states

def tetrahedron_formation(rng=None):
    """
    6-satellite stacked tetrahedral formation. 
    Two stacked tetrahedra of 3 satellites each, offset in z.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    r = 2000.0   # m
    h = 1500.0   # z-stack offset
    states = np.zeros((6, 6))
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    for k, theta in enumerate(angles):
        states[k, 0] = r * np.cos(theta)   # x
        states[k, 2] = r * np.sin(theta)   # y (along-track)
        states[k, 4] = -h/2                # z lower layer
    for k, theta in enumerate(angles):
        states[k+3, 0] = r * np.cos(theta + np.pi/3)
        states[k+3, 2] = r * np.sin(theta + np.pi/3)
        states[k+3, 4] = h/2               # z upper layer
    return states

def add_od_error(states: np.ndarray, sigma_pos_m: float, 
                  sigma_vel_ms: float, rng=None):
    """Inject orbital determination uncertainty (Gaussian noise)."""
    if rng is None:
        rng = np.random.default_rng(99)
    noisy = states.copy()
    N = states.shape[0]
    noisy[:, :3]  += rng.normal(0, sigma_pos_m, (N, 3))
    noisy[:, 3:]  += rng.normal(0, sigma_vel_ms, (N, 3))
    return noisy

print("Formation initializer loaded successfully.")
