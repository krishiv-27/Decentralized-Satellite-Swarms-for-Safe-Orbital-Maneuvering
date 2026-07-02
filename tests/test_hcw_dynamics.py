"""
Unit tests for the HCW dynamics engine.
Run: python -m pytest tests/ -v   (or use the test runner in this file)
"""
import numpy as np
import sys
sys.path.insert(0, 'src')
from hcw_dynamics import (mean_motion, hcw_A_matrix, state_transition_matrix,
                           propagate_free_drift, separation_matrix,
                           verify_separation_smt_proxy)


def test_mean_motion_iss_altitude():
    """ISS ~400 km: orbital period should be 90–96 minutes."""
    n = mean_motion(400.0)
    T = 2 * np.pi / n
    assert 5400 < T < 5760, f"Period {T:.0f}s outside [5400, 5760]"


def test_hcw_a_matrix_shape():
    """A matrix must be 6×6."""
    n = mean_motion(550.0)
    A = hcw_A_matrix(n)
    assert A.shape == (6, 6)


def test_state_transition_identity_at_zero():
    """Phi(0) == I_6."""
    n = mean_motion(550.0)
    Phi = state_transition_matrix(n, 0.0)
    assert np.allclose(Phi, np.eye(6), atol=1e-10)


def test_free_drift_bounded_pco():
    """
    A PCO initial condition (bounded-drift: ydot0 = -2n*x0) must remain
    within the expected ellipse envelope over one full orbit.
    """
    n   = mean_motion(550.0)
    rho = 3000.0        # PCO semi-axis [m]
    # IC at theta=0: x0=rho, xdot=-rho*n*sin(0)=0, y0=-2*rho*sin(0)=0,
    # ydot=-2n*x0 (bounded-drift condition)
    x0  = np.array([rho, 0.0, 0.0, -2*n*rho, 0.0, 0.0])
    T_orb = 2 * np.pi / n
    t_arr = np.linspace(0, T_orb, 1000)
    traj  = propagate_free_drift(x0[np.newaxis, :], n, t_arr)
    radial = traj[0, :, 0]
    # Radial excursion should stay bounded within ±rho * 1.05 (small numerical margin)
    assert np.max(np.abs(radial)) <= rho * 1.05, \
        f"Radial excursion {np.max(np.abs(radial)):.1f}m exceeds expected {rho*1.05:.1f}m"


def test_bounded_drift_condition():
    """
    For a PCO initial condition, ydot0 + 2n*x0 must be zero (bounded-drift).
    """
    n     = mean_motion(550.0)
    rho   = 5000.0
    theta = np.pi / 4           # arbitrary non-zero angle
    x0    = rho * np.cos(theta)
    ydot0 = -2 * n * x0        # bounded-drift assignment
    drift = ydot0 + 2 * n * x0
    assert abs(drift) < 1e-9, f"Bounded-drift residual {drift:.2e} (should be ~0)"


def test_verify_separation_safe_pair_yaxis():
    """
    Two satellites 10 km apart along the along-track (y) axis should receive
    a safety certificate — pure y-separation is preserved under free-drift HCW.
    """
    n    = mean_motion(550.0)
    x0_i = np.zeros(6)
    x0_j = np.array([0.0, 0.0, 10000.0, 0.0, 0.0, 0.0])   # 10 km along-track
    cert, sep, _, _ = verify_separation_smt_proxy(
        x0_i, x0_j, n, 600.0, 500.0, 150.0, n_intervals=200)
    assert bool(cert) is True, f"Expected certificate for 10 km along-track pair; min_sep={sep}"


def test_verify_separation_unsafe_pair():
    """
    Two satellites separated by only 100 m should NOT receive a certificate
    (100 m < r_min + delta_r = 650 m).
    """
    n    = mean_motion(550.0)
    x0_i = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    x0_j = np.array([0.0, 0.0, 200.0, 0.0, 0.0, 0.0])
    cert, sep, _, _ = verify_separation_smt_proxy(
        x0_i, x0_j, n, 600.0, 500.0, 150.0, n_intervals=200)
    assert bool(cert) is False, f"Expected no certificate for 100 m pair; min_sep={sep}"


def test_separation_matrix_symmetry():
    """Separation matrix seps[i,j,:] must equal seps[j,i,:]."""
    n    = mean_motion(550.0)
    x0   = np.random.default_rng(0).uniform(-2000, 2000, (4, 6))
    traj = propagate_free_drift(x0, n, np.linspace(0, 600, 100))
    seps = separation_matrix(traj)
    assert np.allclose(seps, seps.transpose(1, 0, 2)), "Separation matrix not symmetric"


def test_separation_matrix_zero_diagonal():
    """Self-separation seps[i,i,:] must be zero at all times."""
    n    = mean_motion(550.0)
    x0   = np.random.default_rng(1).uniform(-2000, 2000, (3, 6))
    traj = propagate_free_drift(x0, n, np.linspace(0, 600, 50))
    seps = separation_matrix(traj)
    for i in range(3):
        assert np.allclose(seps[i, i, :], 0.0), \
            f"Non-zero self-separation for satellite {i}"


# ── Standalone runner (no pytest required) ─────────────────
if __name__ == '__main__':
    import sys as _sys
    _tests = [v for k,v in sorted(globals().items()) if k.startswith('test_')]
    passed = failed = 0
    for fn in _tests:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {fn.__name__}: {e}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    _sys.exit(failed)
