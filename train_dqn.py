import sys
import os
import math
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from server.sim.world import World, Intersection, Road, Direction
from server.sim.builders import (
    new_world,
    add_intersection,
    add_road,
    wire,
    connect_neighbors,
    spawn_stream,
    PHASES_NS_EW,
)
from server.sim.engine import tick, phase_serves

# ---------------------------------------------------------------------------
# 1. GRID BUILDER -- 4x4 grid of intersections
# ---------------------------------------------------------------------------

GRID_ROWS = 4
GRID_COLS = 4
ROAD_LEN = 8
SOURCE_LEN = 6
SINK_LEN = 5
MIN_GREEN = 6
MAX_GREEN = 45

def iid(r, c):
    return f"I_{r}_{c}"

def build_grid_world(seed: int, horizon: int = 400, dqn_controlled: bool = True) -> World:
    w = new_world("grid_4x4", horizon=horizon, seed=seed, interventions_budget=0)

    # when DQN controls, set phase limits absurdly high so the built-in
    # controller never switches -- the DQN applies its own actions before ticks
    if dqn_controlled:
        _min_p, _max_p = 999999, 999999
    else:
        _min_p, _max_p = MIN_GREEN, MAX_GREEN

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            add_intersection(w, iid(r, c), position=(c, r),
                             min_phase_ticks=_min_p, max_phase_ticks=_max_p)

    # horizontal roads (west-to-east): approach direction is W
    for r in range(GRID_ROWS):
        add_road(w, f"R_src_W_{r}", f"SRC_W_{r}", iid(r, 0), approach="W", length=SOURCE_LEN)
        for c in range(GRID_COLS - 1):
            add_road(w, f"R_h_{r}_{c}", iid(r, c), iid(r, c+1), approach="W", length=ROAD_LEN)
        add_road(w, f"R_sink_E_{r}", iid(r, GRID_COLS-1), f"SINK_E_{r}", approach="W", length=SINK_LEN)

    # vertical roads (south-to-north): approach direction is S
    for c in range(GRID_COLS):
        add_road(w, f"R_src_S_{c}", f"SRC_S_{c}", iid(0, c), approach="S", length=SOURCE_LEN)
        for r in range(GRID_ROWS - 1):
            add_road(w, f"R_v_{r}_{c}", iid(r, c), iid(r+1, c), approach="S", length=ROAD_LEN)
        add_road(w, f"R_sink_N_{c}", iid(GRID_ROWS-1, c), f"SINK_N_{c}", approach="S", length=SINK_LEN)

    # wire each intersection
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            inc = {}
            out = {}
            if c == 0:
                inc["W"] = f"R_src_W_{r}"
            else:
                inc["W"] = f"R_h_{r}_{c-1}"
            if c == GRID_COLS - 1:
                out["E"] = f"R_sink_E_{r}"
            else:
                out["E"] = f"R_h_{r}_{c}"
            if r == 0:
                inc["S"] = f"R_src_S_{c}"
            else:
                inc["S"] = f"R_v_{r-1}_{c}"
            if r == GRID_ROWS - 1:
                out["N"] = f"R_sink_N_{c}"
            else:
                out["N"] = f"R_v_{r}_{c}"
            wire(w, iid(r, c), incoming=inc, outgoing=out)

    connect_neighbors(w)

    # spawn horizontal arterials
    for r in range(GRID_ROWS):
        route = [f"R_src_W_{r}"]
        for c in range(GRID_COLS - 1):
            route.append(f"R_h_{r}_{c}")
        route.append(f"R_sink_E_{r}")
        spawn_stream(w, start_tick=2, end_tick=horizon - 30, period=6,
                     vid_prefix=f"H_{r}", vtype="civilian", route=route, jitter=0.3)

    # spawn vertical arterials
    for c in range(GRID_COLS):
        route = [f"R_src_S_{c}"]
        for r in range(GRID_ROWS - 1):
            route.append(f"R_v_{r}_{c}")
        route.append(f"R_sink_N_{c}")
        spawn_stream(w, start_tick=4, end_tick=horizon - 30, period=8,
                     vid_prefix=f"V_{c}", vtype="civilian", route=route, jitter=0.3)

    return w

# ---------------------------------------------------------------------------
# 2. STATE EXTRACTION -- per-intersection feature vector
# ---------------------------------------------------------------------------
# Features per intersection (14 dims):
#   [0]  queue on S incoming / road_len   (0..1)
#   [1]  queue on W incoming / road_len   (0..1)
#   [2]  occupancy on S incoming / len    (0..1, total vehicles not just tail)
#   [3]  occupancy on W incoming / len    (0..1)
#   [4]  current_phase_is_NS              (0 or 1)
#   [5]  phase_timer / MAX_GREEN          (0..~1, clamped)
#   [6]  can_switch                       (1 if phase_timer >= MIN_GREEN, else 0)
#   [7]  neighbor mean queue S            (0..1)
#   [8]  neighbor mean queue W            (0..1)
#   [9]  downstream E occupancy / len     (0..1, backpressure)
#   [10] downstream N occupancy / len     (0..1, backpressure)
#   [11] total_incoming_occ / capacity    (0..1)
#   [12] S_pressure - W_pressure          (-1..1, normalized)
#   [13] tick / horizon                   (0..1)

STATE_DIM = 14

def get_intersection_ids(world: World) -> list[str]:
    ids = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ids.append(iid(r, c))
    return ids

def queue_norm(world: World, rid: str) -> float:
    if rid not in world.roads:
        return 0.0
    r = world.roads[rid]
    return min(1.0, r.queue_at_tail() / max(1, r.length))

def occ_norm(world: World, rid: str) -> float:
    if rid not in world.roads:
        return 0.0
    r = world.roads[rid]
    return min(1.0, r.occupancy() / max(1, r.length))

def extract_state(world: World, inter_id: str) -> np.ndarray:
    I = world.intersections[inter_id]
    s = np.zeros(STATE_DIM, dtype=np.float32)

    # incoming queue lengths (only S and W have incoming in our grid)
    s_rid = I.incoming.get("S")
    w_rid = I.incoming.get("W")
    s[0] = queue_norm(world, s_rid) if s_rid else 0.0
    s[1] = queue_norm(world, w_rid) if w_rid else 0.0
    s[2] = occ_norm(world, s_rid) if s_rid else 0.0
    s[3] = occ_norm(world, w_rid) if w_rid else 0.0

    # current phase: 1 = NS green (serves S approach), 0 = EW green (serves W approach)
    phase = I.current_phase()
    s[4] = 1.0 if frozenset({"N", "S"}) == phase else 0.0

    # phase timer (use real constants, not the hacked ones)
    s[5] = min(1.0, I.phase_timer / MAX_GREEN)
    s[6] = 1.0 if I.phase_timer >= MIN_GREEN else 0.0

    # neighbor average queues
    s_neighbors = []
    w_neighbors = []
    for nid in I.neighbors:
        if nid not in world.intersections:
            continue
        NI = world.intersections[nid]
        n_s_rid = NI.incoming.get("S")
        n_w_rid = NI.incoming.get("W")
        if n_s_rid:
            s_neighbors.append(queue_norm(world, n_s_rid))
        if n_w_rid:
            w_neighbors.append(queue_norm(world, n_w_rid))
    s[7] = np.mean(s_neighbors) if s_neighbors else 0.0
    s[8] = np.mean(w_neighbors) if w_neighbors else 0.0

    # downstream backpressure (outgoing roads)
    e_rid = I.outgoing.get("E")
    n_rid = I.outgoing.get("N")
    s[9] = occ_norm(world, e_rid) if e_rid else 0.0
    s[10] = occ_norm(world, n_rid) if n_rid else 0.0

    # total incoming load
    total_occ = 0.0
    total_cap = 0.0
    for d in ["S", "W"]:
        rid = I.incoming.get(d)
        if rid and rid in world.roads:
            total_occ += world.roads[rid].occupancy()
            total_cap += world.roads[rid].length
    s[11] = total_occ / max(1, total_cap)

    # pressure difference: S demand vs W demand (NS_pressure - EW_pressure)
    s_q = world.roads[s_rid].queue_at_tail() if s_rid and s_rid in world.roads else 0
    w_q = world.roads[w_rid].queue_at_tail() if w_rid and w_rid in world.roads else 0
    total_q = max(1.0, float(s_q + w_q))
    s[12] = (s_q - w_q) / total_q

    # time progress
    s[13] = world.tick / max(1, world.horizon)

    return s

def extract_all_states(world: World, inter_ids: list[str]) -> np.ndarray:
    return np.stack([extract_state(world, iid) for iid in inter_ids])

# ---------------------------------------------------------------------------
# 3. ACTION SPACE -- binary: 0 = hold current phase, 1 = switch
# ---------------------------------------------------------------------------

NUM_ACTIONS = 2

def apply_dqn_actions(world: World, inter_ids: list[str], actions: np.ndarray):
    for i, inter_id in enumerate(inter_ids):
        I = world.intersections[inter_id]
        if I.preempt_direction is not None:
            continue
        if actions[i] == 1 and I.phase_timer >= MIN_GREEN:
            I.current_phase_idx = (I.current_phase_idx + 1) % len(I.phases)
            I.phase_timer = 0

# ---------------------------------------------------------------------------
# 4. REWARD -- per-intersection local reward
# ---------------------------------------------------------------------------

def compute_local_rewards(
    world: World,
    inter_ids: list[str],
    prev_queues: dict[str, dict[str, int]],
) -> np.ndarray:
    rewards = np.zeros(len(inter_ids), dtype=np.float32)
    for i, inter_id in enumerate(inter_ids):
        I = world.intersections[inter_id]
        r = 0.0

        cur_queues = {}
        total_queue = 0
        for d in ["S", "W"]:
            rid = I.incoming.get(d)
            if rid and rid in world.roads:
                q = world.roads[rid].queue_at_tail()
                cur_queues[d] = q
                total_queue += q
            else:
                cur_queues[d] = 0

        prev_q = prev_queues.get(inter_id, {})
        prev_total = sum(prev_q.get(d, 0) for d in ["S", "W"])

        # throughput reward: queues went down = vehicles moved through
        queue_delta = prev_total - total_queue
        r += 0.5 * queue_delta

        # standing queue penalty
        r -= 0.03 * total_queue

        # wasted green penalty
        phase = I.current_phase()
        served_demand = 0
        for d in phase:
            rid = I.incoming.get(d)
            if rid and rid in world.roads:
                served_demand += world.roads[rid].queue_at_tail()
        unserved_demand = 0
        for d in ["S", "W"]:
            if d not in phase:
                rid = I.incoming.get(d)
                if rid and rid in world.roads:
                    unserved_demand += world.roads[rid].queue_at_tail()
        if served_demand == 0 and unserved_demand > 0:
            r -= 0.4

        # balance penalty
        s_q = cur_queues.get("S", 0)
        w_q = cur_queues.get("W", 0)
        if s_q + w_q > 0:
            imbalance = abs(s_q - w_q) / (s_q + w_q)
            r -= 0.08 * imbalance

        rewards[i] = r
    return rewards

def snapshot_queues(world: World, inter_ids: list[str]) -> dict[str, dict[str, int]]:
    out = {}
    for inter_id in inter_ids:
        I = world.intersections[inter_id]
        q = {}
        for d in ["S", "W"]:
            rid = I.incoming.get(d)
            if rid and rid in world.roads:
                q[d] = world.roads[rid].queue_at_tail()
            else:
                q[d] = 0
        out[inter_id] = q
    return out

# ---------------------------------------------------------------------------
# 5. DQN NETWORK -- small, fast on CPU
# ---------------------------------------------------------------------------
# 14 -> 64 -> 64 -> 2
# total params: 14*64 + 64 + 64*64 + 64 + 64*2 + 2 = 5,186
# 16 forward passes per tick: ~0.1ms on CPU, negligible

class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=NUM_ACTIONS, hidden=64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=-1, keepdim=True))

# ---------------------------------------------------------------------------
# 6. REPLAY BUFFER
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)

# ---------------------------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------------------------

NUM_EPISODES = 150
DECISION_INTERVAL = 5    # agent picks action every 5 ticks (~matches MIN_GREEN)
GAMMA = 0.95
LR = 3e-4
BATCH_SIZE = 128
BUFFER_SIZE = 80000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 60           # episodes over which epsilon decays
TARGET_UPDATE = 5        # copy policy -> target every N episodes
MIN_BUFFER = 512
TRAIN_ITERS_PER_STEP = 1


def train():
    device = torch.device("cpu")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    episode_rewards = []
    t_start = time.time()

    for ep in range(NUM_EPISODES):
        seed = ep * 7 + 42
        horizon = 300 + random.randint(0, 100)
        world = build_grid_world(seed=seed, horizon=horizon, dqn_controlled=True)
        inter_ids = get_intersection_ids(world)
        n_inters = len(inter_ids)

        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-ep / EPS_DECAY)

        states = extract_all_states(world, inter_ids)
        prev_queues = snapshot_queues(world, inter_ids)
        ep_reward = 0.0
        steps = 0

        while world.tick < world.horizon:
            # epsilon-greedy action selection for all 16 intersections at once
            if random.random() < eps:
                actions = np.random.randint(0, NUM_ACTIONS, size=n_inters)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(states, dtype=torch.float32, device=device))
                    actions = q_vals.argmax(dim=1).numpy()

            # apply DQN decisions before ticking
            apply_dqn_actions(world, inter_ids, actions)

            # run simulation for DECISION_INTERVAL ticks
            # built-in controller is disabled (min/max phase = 999999)
            for _ in range(DECISION_INTERVAL):
                tick(world)
                if world.tick >= world.horizon:
                    break

            next_states = extract_all_states(world, inter_ids)
            rewards = compute_local_rewards(world, inter_ids, prev_queues)
            done = world.tick >= world.horizon

            # store each intersection's transition independently (parameter sharing)
            for i in range(n_inters):
                buffer.push(states[i], actions[i], rewards[i], next_states[i], float(done))

            ep_reward += rewards.sum()
            states = next_states
            prev_queues = snapshot_queues(world, inter_ids)
            steps += 1

            if len(buffer) >= MIN_BUFFER:
                for _ in range(TRAIN_ITERS_PER_STEP):
                    _train_step(policy_net, target_net, optimizer, buffer, device)

        episode_rewards.append(ep_reward)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        elapsed = time.time() - t_start
        avg_r = np.mean(episode_rewards[-10:])
        print(f"ep {ep+1}/{NUM_EPISODES}  steps={steps}  reward={ep_reward:.1f}  "
              f"avg10={avg_r:.1f}  eps={eps:.3f}  elapsed={elapsed:.0f}s")

        if elapsed > 540:
            print("approaching 10 min limit, stopping early")
            break

    save_path = os.path.join(os.path.dirname(__file__), "dqn_traffic.pt")
    torch.save({
        "state_dict": policy_net.state_dict(),
        "state_dim": STATE_DIM,
        "n_actions": NUM_ACTIONS,
        "hidden": 64,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "min_green": MIN_GREEN,
        "decision_interval": DECISION_INTERVAL,
    }, save_path)
    print(f"saved model to {save_path}")
    print(f"total training time: {time.time() - t_start:.1f}s")
    print(f"final avg reward (last 10 eps): {np.mean(episode_rewards[-10:]):.1f}")


def _train_step(policy_net, target_net, optimizer, buffer, device):
    s, a, r, ns, d = buffer.sample(BATCH_SIZE)
    s_t = torch.tensor(s, device=device)
    a_t = torch.tensor(a, device=device).unsqueeze(1)
    r_t = torch.tensor(r, device=device)
    ns_t = torch.tensor(ns, device=device)
    d_t = torch.tensor(d, device=device)

    q_values = policy_net(s_t).gather(1, a_t).squeeze(1)
    with torch.no_grad():
        next_q = target_net(ns_t).max(dim=1)[0]
        target = r_t + GAMMA * next_q * (1 - d_t)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

# ---------------------------------------------------------------------------
# 8. INFERENCE HELPER -- load model and run at serve time
# ---------------------------------------------------------------------------

def load_dqn(path: str, device="cpu") -> DQN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = DQN(
        state_dim=ckpt["state_dim"],
        n_actions=ckpt["n_actions"],
        hidden=ckpt["hidden"],
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net

def dqn_select_actions(net: DQN, world: World, inter_ids: list[str]) -> np.ndarray:
    states = extract_all_states(world, inter_ids)
    with torch.no_grad():
        q_vals = net(torch.tensor(states, dtype=torch.float32))
        return q_vals.argmax(dim=1).numpy()


if __name__ == "__main__":
    train()
