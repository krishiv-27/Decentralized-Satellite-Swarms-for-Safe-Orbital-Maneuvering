"""
Unit tests for the swarm protocol and Byzantine consensus.
Run: python -m pytest tests/ -v
"""
import numpy as np
import sys
sys.path.insert(0, 'src')
from swarm_protocol import SwarmProtocol, SatState, ManeuverIntent
from hcw_dynamics import mean_motion
from formation_initializer import pco_formation, add_od_error


N_ORB = mean_motion(550.0)


def make_proto(N, f=0, dropout=0.0, seed=0):
    rng = np.random.default_rng(42)
    x0  = pco_formation(N, 5000, n=N_ORB, rng=rng)
    x0  = add_od_error(x0, 30.0, 0.05, rng=rng)
    p   = SwarmProtocol(N=N, f_byzantine=f, n_orbital=N_ORB,
                        r_min=500.0, delta_r=150.0,
                        T_max_liveness=30000.0,
                        link_dropout_rate=dropout, seed=seed)
    p.initialize_agents(x0, byzantine_ids=list(range(f)))
    return p, x0


def test_quorum_formula():
    """Quorum must satisfy q > (N+f)/2."""
    for N in [6, 12, 24]:
        for f in range((N-1)//3 + 1):
            p, _ = make_proto(N, f)
            assert p.quorum > (N + f) / 2, f"Quorum too low: N={N} f={f} q={p.quorum}"


def test_p1_no_simultaneous_burns():
    """Protocol must not authorize two satellites in the same conjunction pair."""
    p, x0 = make_proto(6, f=0)
    # Force a conjunction between sats 0 and 1
    p.conjunction_pairs.add((0, 1))
    p.agents[0].conjunction_detected = True
    p.agents[1].conjunction_detected = True

    # Both want to maneuver with certified intents
    intents = {
        0: ManeuverIntent(0, np.zeros(3), 30.0, 600.0, True, 0.0),
        1: ManeuverIntent(1, np.zeros(3), 30.0, 600.0, True, 0.0),
    }
    import time
    G    = p.build_topology()
    auth = p.run_consensus_round(G, intents, time.perf_counter())

    p1_viol = p.check_p1_mutual_exclusion(auth)
    assert p1_viol == 0, f"P1 violated: both sats 0 and 1 authorized"


def test_byzantine_node_does_not_get_authorized():
    """Byzantine nodes must never receive maneuver authorization."""
    p, x0 = make_proto(6, f=1)  # sat 0 is Byzantine
    curr = np.array([[np.linalg.norm(x0[i,:3]-x0[j,:3]) if i!=j else 0.0
                      for j in range(6)] for i in range(6)])
    p.detect_conjunctions(curr, 8000.0)

    import time
    G       = p.build_topology()
    intents = {0: ManeuverIntent(0, np.zeros(3), 30.0, 600.0, False, 0.0)}
    auth    = p.run_consensus_round(G, intents, time.perf_counter())

    # Byzantine sat 0 should not be in authorized (no certified intent)
    assert 0 not in auth or not auth.get(0, False), \
        "Byzantine satellite was authorized to maneuver"


def test_topology_respects_dropout():
    """With 100% dropout, graph should have no edges."""
    p, _ = make_proto(6, dropout=1.0, seed=99)
    G = p.build_topology()
    assert G.number_of_edges() == 0, "Full dropout should produce empty graph"


def test_topology_full_connectivity():
    """With 0% dropout, all N*(N-1) directed edges should exist."""
    N = 6
    p, _ = make_proto(N, dropout=0.0, seed=0)
    G = p.build_topology()
    assert G.number_of_edges() == N * (N-1), \
        f"Expected {N*(N-1)} edges, got {G.number_of_edges()}"


def test_p4_termination():
    """All non-Byzantine agents must eventually reach COAST or FREE_DRIFT."""
    p, x0 = make_proto(6, f=1)
    # Manually advance all non-Byzantine agents to COAST
    for a in p.agents:
        if not a.is_byzantine:
            a.state = SatState.POST_BURN_COAST
    assert p.check_p4_termination() == 0, "P4 violated after manual COAST assignment"
