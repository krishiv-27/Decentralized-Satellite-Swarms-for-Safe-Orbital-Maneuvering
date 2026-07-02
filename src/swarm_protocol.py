"""
Distributed Swarm Collision Avoidance Protocol Simulator
Implements: Byzantine-resilient consensus, LTL property monitor,
            communication topology with dropouts, maneuver arbitration
"""
import numpy as np
import networkx as nx
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Tuple

class SatState(Enum):
    FREE_DRIFT        = auto()
    COMPUTING         = auto()
    MANEUVER_COMMITTED = auto()
    EXECUTING         = auto()
    POST_BURN_COAST   = auto()

@dataclass
class ManeuverIntent:
    sat_id: int
    delta_v: np.ndarray       # (3,) m/s
    burn_start: float         # seconds from now
    burn_duration: float      # seconds
    safety_certified: bool    # SMT oracle result
    cert_time_ms: float       # oracle latency

@dataclass 
class SatelliteAgent:
    sat_id: int
    state: SatState = SatState.FREE_DRIFT
    x0: np.ndarray = field(default_factory=lambda: np.zeros(6))
    intents_received: Dict[int, ManeuverIntent] = field(default_factory=dict)
    is_byzantine: bool = False
    maneuver_locked: bool = False
    conjunction_detected: bool = False
    maneuver_latency: Optional[float] = None   # time from detection to commit
    
    # LTL property trace
    safe_sep_violated: bool = False
    deadlocked: bool = False
    liveness_timeout: bool = False

class SwarmProtocol:
    """
    Simulates N-satellite swarm with Byzantine consensus and
    LTL property monitoring over P1–P4.
    """
    def __init__(self, N: int, f_byzantine: int, n_orbital: float,
                 r_min: float, delta_r: float, T_max_liveness: float,
                 link_dropout_rate: float = 0.0, seed: int = 42):
        self.N = N
        self.f = f_byzantine
        self.n = n_orbital
        self.r_min = r_min
        self.delta_r = delta_r
        self.T_max = T_max_liveness
        self.dropout = link_dropout_rate
        self.rng = np.random.default_rng(seed)
        
        # Quorum threshold for Byzantine-resilient consensus
        # Must satisfy: q > (N + f) / 2  (Byzantine majority)
        self.quorum = int(np.ceil((N + f_byzantine + 1) / 2))
        
        self.agents: List[SatelliteAgent] = []
        self.conjunction_pairs: Set[Tuple[int,int]] = set()
        self.time = 0.0
        self.smt_calls = 0
        self.smt_total_ms = 0.0
        
        # LTL property counters
        self.p1_violations = 0   # mutual exclusion broken
        self.p2_violations = 0   # separation violated during execution
        self.p3_violations = 0   # liveness timeout
        self.p4_violations = 0   # termination failure
        
    def initialize_agents(self, initial_states: np.ndarray,
                          byzantine_ids: List[int]):
        for i in range(self.N):
            agent = SatelliteAgent(
                sat_id=i,
                x0=initial_states[i].copy(),
                is_byzantine=(i in byzantine_ids)
            )
            self.agents.append(agent)
    
    def build_topology(self) -> nx.DiGraph:
        """
        Build communication graph with dropout.
        Default: fully connected minus random dropped edges.
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if self.rng.random() > self.dropout:
                        G.add_edge(i, j)
        return G
    
    def detect_conjunctions(self, seps: np.ndarray, warn_dist: float):
        """Flag all satellite pairs below warning distance."""
        self.conjunction_pairs.clear()
        for i in range(self.N):
            for j in range(i+1, self.N):
                if seps[i,j] < warn_dist:
                    self.conjunction_pairs.add((i,j))
                    self.agents[i].conjunction_detected = True
                    self.agents[j].conjunction_detected = True
    
    def run_consensus_round(self, G: nx.DiGraph, 
                            certified_intents: Dict[int, ManeuverIntent],
                            detect_time: float) -> Dict[int, bool]:
        """
        Run one round of Byzantine-resilient maneuver consensus.
        Returns: dict mapping sat_id -> authorized_to_maneuver
        """
        authorized = {}
        
        # Phase 1: Broadcast intents to reachable neighbors
        # Byzantine satellites broadcast random/conflicting intents
        intent_votes: Dict[int, List[ManeuverIntent]] = {i: [] for i in range(self.N)}
        
        for sender_id, intent in certified_intents.items():
            sender = self.agents[sender_id]
            neighbors = list(G.successors(sender_id))
            
            for nb in neighbors:
                if sender.is_byzantine:
                    # Byzantine: send corrupted intent (random delta-v)
                    fake_intent = ManeuverIntent(
                        sat_id=sender_id,
                        delta_v=self.rng.uniform(-0.1, 0.1, 3),
                        burn_start=intent.burn_start,
                        burn_duration=intent.burn_duration,
                        safety_certified=False,
                        cert_time_ms=0.0
                    )
                    intent_votes[nb].append(fake_intent)
                else:
                    intent_votes[nb].append(intent)
        
        # Phase 2: Each non-Byzantine sat counts consistent votes
        for i in range(self.N):
            agent = self.agents[i]
            if agent.is_byzantine:
                continue
            
            votes_for_i = [v for v in intent_votes[i] if v.sat_id == i 
                           and v.safety_certified]
            
            if len(votes_for_i) >= self.quorum - 1:  # -1: self-vote
                agent.state = SatState.MANEUVER_COMMITTED
                agent.maneuver_locked = True
                commit_time = time.perf_counter()
                agent.maneuver_latency = (commit_time - detect_time) * 1000
                authorized[i] = True
            else:
                authorized[i] = False
        
        return authorized
    
    def check_p1_mutual_exclusion(self, authorized: Dict[int, bool]) -> int:
        """P1: No two satellites in same conjunction pair both execute simultaneously."""
        violations = 0
        for (i, j) in self.conjunction_pairs:
            if authorized.get(i, False) and authorized.get(j, False):
                violations += 1
                self.agents[i].safe_sep_violated = True
                self.agents[j].safe_sep_violated = True
        return violations
    
    def check_p3_liveness(self, latencies: List[float]) -> int:
        """P3: All conjunction-flagged satellites must commit within T_max ms."""
        violations = 0
        for a in self.agents:
            if a.conjunction_detected and not a.is_byzantine:
                if a.maneuver_latency is None or a.maneuver_latency > self.T_max:
                    violations += 1
                    a.liveness_timeout = True
        return violations
    
    def check_p4_termination(self) -> int:
        """P4: All non-Byzantine agents must eventually reach COAST/FREE_DRIFT."""
        violations = 0
        for a in self.agents:
            if not a.is_byzantine:
                if a.state not in (SatState.POST_BURN_COAST, SatState.FREE_DRIFT):
                    violations += 1
                    a.deadlocked = True
        return violations

print("Swarm protocol simulator loaded successfully.")
