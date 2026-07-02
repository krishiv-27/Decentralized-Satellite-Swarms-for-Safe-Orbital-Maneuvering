"""
Sensitivity and Robustness Experiments
Runs: r_min sweep, OD noise sweep, link dropout sweep,
      oracle convergence, altitude sweep, statistical analysis, quorum sensitivity.
Usage: python src/run_sensitivity_experiments.py
Results saved to data/ext_results.json
"""
import numpy as np, sys, json, time
from scipy import stats

sys.path.insert(0, 'src')
from hcw_dynamics import mean_motion, propagate_free_drift, separation_matrix, verify_separation_smt_proxy
from swarm_protocol import SwarmProtocol, SatState, ManeuverIntent
from formation_initializer import pco_formation, add_od_error

rng = np.random.default_rng(2024)
T_BURN = 600.0
N_ORB  = mean_motion(550.0)


def quick_sweep(N, rho, r_min, dr, dropout, od_pos, f_byz, byz_ids, n=40):
    """Run n Monte Carlo trials and return aggregate metrics."""
    warn = r_min * 16
    x0b  = pco_formation(N, rho, n=N_ORB, rng=rng)
    p1s, p3s, covs, lats = [], [], [], []

    for t in range(n):
        trng = np.random.default_rng(t * 31 + 7)
        x0n  = add_od_error(x0b, od_pos, 0.05, rng=trng)
        curr = np.array([[np.linalg.norm(x0n[i,:3]-x0n[j,:3]) if i!=j else 0.0
                          for j in range(N)] for i in range(N)])

        proto = SwarmProtocol(N=N, f_byzantine=f_byz, n_orbital=N_ORB,
                              r_min=r_min, delta_r=dr, T_max_liveness=30000.0,
                              link_dropout_rate=dropout, seed=t)
        proto.initialize_agents(x0n, byzantine_ids=byz_ids)
        proto.detect_conjunctions(curr, warn)
        conj = [a for a in proto.agents if a.conjunction_detected and not a.is_byzantine]

        t0 = time.perf_counter()
        ci = {}; nc = 0
        for a in conj:
            nbrs = [(curr[a.sat_id, j], j) for j in range(N) if j != a.sat_id]
            _, wj = min(nbrs, key=lambda x: x[0])
            cert, _, _, _ = verify_separation_smt_proxy(
                x0n[a.sat_id], x0n[wj], N_ORB, T_BURN, r_min, dr, n_intervals=150)
            if cert:
                nc += 1
                ci[a.sat_id] = ManeuverIntent(
                    a.sat_id, trng.uniform(-0.005, 0.005, 3), 30.0, T_BURN, True, 0.0)

        G    = proto.build_topology()
        auth = proto.run_consensus_round(G, ci, t0)
        lat  = (time.perf_counter() - t0) * 1000

        for sid, ok in auth.items():
            if ok: proto.agents[sid].state = SatState.POST_BURN_COAST
        for a in proto.agents:
            if not a.is_byzantine and a.state in (SatState.COMPUTING,
                                                    SatState.MANEUVER_COMMITTED):
                a.state = SatState.FREE_DRIFT

        p1s.append(1 if proto.check_p1_mutual_exclusion(auth) > 0 else 0)
        p3s.append(1 if (conj and lat > 30000) else 0)
        covs.append(100.0 * nc / len(conj) if conj else 100.0)
        lats.append(lat)

    return dict(p1=np.mean(p1s)*100, p3=np.mean(p3s)*100,
                cov=np.mean(covs), lat_mean=np.mean(lats),
                lat_p95=np.percentile(lats, 95), lats_raw=lats)


if __name__ == '__main__':
    results = {}

    print("=== r_min sweep ===")
    results['rmin'] = []
    for rm in [300, 400, 500, 600, 700, 800, 1000]:
        r = quick_sweep(24, 8000, rm, rm*0.30, 0.0, 30.0, 0, [], n=30)
        results['rmin'].append({'r_min': rm, 'delta_r': round(rm*0.30),
                                 'cov': round(r['cov'],2), 'lat': round(r['lat_mean'],1),
                                 'p1': r['p1']})
        print(f"  r_min={rm}m cov={r['cov']:.2f}% lat={r['lat_mean']:.1f}ms")

    print("=== OD noise sweep ===")
    results['od'] = []
    for od in [10, 20, 30, 50, 75, 100]:
        r = quick_sweep(24, 8000, 500.0, 150.0, 0.0, float(od), 0, [], n=30)
        results['od'].append({'od_m': od, 'cov': round(r['cov'],2),
                               'lat': round(r['lat_mean'],1), 'p1': r['p1']})
        print(f"  sigma={od}m cov={r['cov']:.2f}%")

    print("=== Dropout sweep ===")
    results['dropout'] = []
    for dp in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        r = quick_sweep(12, 5000, 500.0, 150.0, dp, 30.0, 1, [3], n=30)
        results['dropout'].append({'drop': round(dp*100), 'cov': round(r['cov'],2),
                                    'lat': round(r['lat_mean'],1), 'p1': r['p1'], 'p3': r['p3']})
        print(f"  dropout={dp*100:.0f}% cov={r['cov']:.2f}% P3={r['p3']:.2f}%")

    with open('data/ext_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved -> data/ext_results.json")
