import os
from typing import Optional

import numpy as np

from .world import World, Intersection

STATE_DIM = 14
MIN_GREEN = 6
DECISION_INTERVAL = 5


class DQNController:
    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.feat_w1 = data["feature.0.weight"]
        self.feat_b1 = data["feature.0.bias"]
        self.feat_w2 = data["feature.2.weight"]
        self.feat_b2 = data["feature.2.bias"]
        self.val_w1 = data["value.0.weight"]
        self.val_b1 = data["value.0.bias"]
        self.val_w2 = data["value.2.weight"]
        self.val_b2 = data["value.2.bias"]
        self.adv_w1 = data["advantage.0.weight"]
        self.adv_b1 = data["advantage.0.bias"]
        self.adv_w2 = data["advantage.2.weight"]
        self.adv_b2 = data["advantage.2.bias"]

    def forward(self, states: np.ndarray) -> np.ndarray:
        x = states @ self.feat_w1.T + self.feat_b1
        x = np.maximum(x, 0)
        x = x @ self.feat_w2.T + self.feat_b2
        x = np.maximum(x, 0)
        v = x @ self.val_w1.T + self.val_b1
        v = np.maximum(v, 0)
        v = v @ self.val_w2.T + self.val_b2
        a = x @ self.adv_w1.T + self.adv_b1
        a = np.maximum(a, 0)
        a = a @ self.adv_w2.T + self.adv_b2
        q = v + (a - a.mean(axis=-1, keepdims=True))
        return q

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        q = self.forward(states)
        return q.argmax(axis=-1)


def _queue_norm(world: World, rid: Optional[str]) -> float:
    if not rid or rid not in world.roads:
        return 0.0
    r = world.roads[rid]
    return min(1.0, r.queue_at_tail() / max(1, r.length))


def _occ_norm(world: World, rid: Optional[str]) -> float:
    if not rid or rid not in world.roads:
        return 0.0
    r = world.roads[rid]
    return min(1.0, r.occupancy() / max(1, r.length))


def extract_state(world: World, I: Intersection) -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s_rid = I.incoming.get("S")
    w_rid = I.incoming.get("W")
    s[0] = _queue_norm(world, s_rid)
    s[1] = _queue_norm(world, w_rid)
    s[2] = _occ_norm(world, s_rid)
    s[3] = _occ_norm(world, w_rid)
    phase = I.current_phase()
    s[4] = 1.0 if frozenset({"N", "S"}) == phase else 0.0
    s[5] = min(1.0, I.phase_timer / 45.0)
    s[6] = 1.0 if I.phase_timer >= MIN_GREEN else 0.0
    s_neighbors = []
    w_neighbors = []
    for nid in I.neighbors:
        NI = world.intersections.get(nid)
        if NI is None:
            continue
        s_neighbors.append(_queue_norm(world, NI.incoming.get("S")))
        w_neighbors.append(_queue_norm(world, NI.incoming.get("W")))
    s[7] = float(np.mean(s_neighbors)) if s_neighbors else 0.0
    s[8] = float(np.mean(w_neighbors)) if w_neighbors else 0.0
    e_rid = I.outgoing.get("E")
    n_rid = I.outgoing.get("N")
    s[9] = _occ_norm(world, e_rid)
    s[10] = _occ_norm(world, n_rid)
    total_occ = 0.0
    total_cap = 0.0
    for d in ["S", "W"]:
        rid = I.incoming.get(d)
        if rid and rid in world.roads:
            total_occ += world.roads[rid].occupancy()
            total_cap += world.roads[rid].length
    s[11] = total_occ / max(1, total_cap)
    s_q = world.roads[s_rid].queue_at_tail() if s_rid and s_rid in world.roads else 0
    w_q = world.roads[w_rid].queue_at_tail() if w_rid and w_rid in world.roads else 0
    s[12] = (s_q - w_q) / max(1.0, float(s_q + w_q))
    s[13] = world.tick / max(1, world.horizon)
    return s


def rl_step(controller: DQNController, world: World) -> None:
    inter_ids = sorted(world.intersections.keys())
    states = np.stack([extract_state(world, world.intersections[iid]) for iid in inter_ids])
    actions = controller.select_actions(states)
    for i, iid in enumerate(inter_ids):
        I = world.intersections[iid]
        if I.preempt_direction is not None:
            continue
        if actions[i] == 1 and I.phase_timer >= MIN_GREEN:
            I.current_phase_idx = (I.current_phase_idx + 1) % len(I.phases)
            I.phase_timer = 0


_controller_cache: Optional[DQNController] = None


def get_controller() -> Optional[DQNController]:
    global _controller_cache
    if _controller_cache is not None:
        return _controller_cache
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "..", "dqn_weights.npz"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "dqn_weights.npz"),
        "/app/env/dqn_weights.npz",
        "dqn_weights.npz",
    ]:
        if os.path.exists(candidate):
            _controller_cache = DQNController(candidate)
            return _controller_cache
    return None
