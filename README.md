---
title: TrafficOps Environment Server
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TrafficOps — LLM-Supervised Adaptive Traffic Signal Control

An OpenEnv environment where an LLM agent supervises a **4x4 grid of 16 intersections**, each running a trained Dueling DQN controller. The LLM acts as a city-level traffic operations supervisor — coordinating corridors, rerouting around incidents, and preempting signals for emergency vehicles.

## Why TrafficOps?

Real-world traffic operations centers rely on human operators to coordinate signal timing across city-wide networks, reroute traffic around accidents, and clear paths for emergency vehicles. Local adaptive controllers (like Max-Pressure or DQN agents) handle routine phase switching, but they can't see the big picture. TrafficOps simulates this hierarchical control paradigm: trained RL agents handle individual intersections, while the LLM agent provides the network-level intelligence that only a supervisor with a city-wide view can deliver.

## Architecture — Three Tiers of Intelligence

```
Tier 3: LLM Supervisor (the player)
  │ Sees: full 4x4 grid state, incidents, emergencies
  │ Decides: bias, preempt, reroute, coordinate
  ▼
Tier 2: Trained Dueling DQN (16 agents, shared weights)
  │ 14-dim state → 64 → 64 → value/advantage → hold/switch
  │ Respects LLM overrides (preempt, bias)
  ▼
Tier 1: 4x4 Grid Simulation
  16 intersections, 40 road segments, cell-automaton physics
```

**Dueling DQN details:** Trained from scratch on this environment for 150 episodes (~51s on CPU). Architecture: 14→64→64→V/A streams (5,186 params, 42KB). Inference is pure numpy (no torch needed at serve time). Inspired by prior MARL research using IQL/VDN/QMIX with GCN/GAT graph encoders.

## Grid Layout

```
SRC_S_0   SRC_S_1   SRC_S_2   SRC_S_3
  ↓         ↓         ↓         ↓
SRC_W_0 → I_0_0 ─── I_0_1 ─── I_0_2 ─── I_0_3 → SINK_E_0
            │         │         │         │
SRC_W_1 → I_1_0 ─── I_1_1 ─── I_1_2 ─── I_1_3 → SINK_E_1
            │         │         │         │
SRC_W_2 → I_2_0 ─── I_2_1 ─── I_2_2 ─── I_2_3 → SINK_E_2
            │         │         │         │
SRC_W_3 → I_3_0 ─── I_3_1 ─── I_3_2 ─── I_3_3 → SINK_E_3
            ↓         ↓         ↓         ↓
         SINK_N_0  SINK_N_1  SINK_N_2  SINK_N_3
```

Road naming: `R_h_{row}_{col}` (horizontal W→E), `R_v_{row}_{col}` (vertical S→N)
Corridors: `ew_row_0..3` (east-west), `ns_col_0..3` (north-south)

## Action Space

| Operation | Description |
|---|---|
| `noop` | Do nothing (let DQN agents handle it) |
| `set_bias` | Increase green-time weight for a direction at intersections |
| `set_coordination` | Synchronize signals along a corridor for a green wave |
| `preempt` | Force green for a direction (emergency clearance) |
| `reroute` | Redirect vehicles around a blocked road |
| `set_policy` | Apply a policy (e.g., school_zone) to reduce phase duration |
| `cancel` | Cancel an active plan |

## Observation Space

Each step returns `TrafficOpsObservation` with:
- **summary** — compact 4x4 grid view with phase/queue at each intersection, detour hints, emergency routes
- **roads** — all 40 road segments: IDs, from/to nodes, lengths, occupancy, queue at stop line, blocked status
- **intersections** — 16 intersections: phase, queues per direction, bias, preempt state, neighbors
- **corridors** — 8 corridors: coordination status, flow, green-wave offsets
- **incidents** — active road blockages with start/end ticks
- **emergencies** — live emergency vehicles with remaining route, ETA, and destination
- **active_plans** — agent's currently active interventions with expiry
- **metrics** — cleared/spawned counts, wait times, wasted green, gridlocks

## Tasks (5, Easy → Expert)

### 1. `grid_balanced` (Easy)
4x4 grid with balanced east-west and north-south traffic. One ambulance crosses the grid mid-episode. Horizon: 250, budget: 6.

### 2. `demand_shift` (Medium)
Heavy north-south traffic flips to heavy east-west mid-episode. DQN agents adapt slowly. LLM must detect the shift and rebias the network. Plus an ambulance during transition. Horizon: 300, budget: 6.

### 3. `incident_corridor` (Hard)
A road blockage cuts row 1 mid-corridor. LLM must reroute traffic around it and preempt for an ambulance and fire truck. DQN agents have no concept of routing. Horizon: 280, budget: 8.

### 4. `rush_hour_wave` (Hard)
Traffic from all southern sources triples suddenly. The demand wave ripples north through the grid. LLM must coordinate north-south corridors and handle a police car during peak. Horizon: 280, budget: 8.

### 5. `multi_crisis` (Expert)
Two incidents block different roads at different times. Three emergency vehicles (ambulance, fire, police) arrive staggered. LLM must triage: which crisis gets budget first? Tests sequential rerouting, multi-emergency preemption, and budget management under overlapping crises. Horizon: 320, budget: 12.

## Reward Design

Dense per-tick reward:
- +1.0 per cleared civilian, +5.0 per cleared emergency
- -0.002 x total waiting ticks across all vehicles
- -0.3 per wasted green tick (serving empty approaches)
- -12.0 per gridlock event

Final episode score uses the openenv **Rubric** framework (`WeightedSum` of 6 `Rubric` subclasses): throughput, emergency clearance, fairness (max wait), efficiency (wasted green), planning quality (budget usage), and safety (gridlock avoidance). Weights vary by task.

## Quick Start — Playing in the Playground

1. Click **Reset** to start an episode
2. Set **Op** to `set_bias`, **Targets** to `I_0_0,I_1_0,I_2_0,I_3_0`, **Params** to `{"direction":"W","multiplier":2.5,"duration_ticks":200}`, click **Step**
3. Click **Step** with Op=`noop` to advance the simulation
4. When you see EMERGENCIES in the response, use `preempt` on intersections along their route
5. When you see INCIDENTS, use `reroute` with a detour path

## Baseline Scores (DQN-only, no LLM intervention)

| Task | Difficulty | DQN Noop | Smart LLM |
|---|---|---|---|
| grid_balanced | Easy | 0.77 | — |
| demand_shift | Medium | 0.71 | — |
| incident_corridor | Hard | 0.59 | 0.70 |
| rush_hour_wave | Hard | 0.60 | — |
| multi_crisis | Expert | 0.67 | — |

Smart LLM agents using bias + preempt + reroute score 0.10-0.15 higher on hard/expert tasks.

## Setup

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

```bash
# Docker
docker build -t trafficops-env:latest .
docker run -p 8000:8000 trafficops-env:latest
```

```bash
# Inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

## Project Structure

```
trafficops/
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── dqn_weights.npz          # Trained Dueling DQN (numpy, no torch needed)
├── inference.py
├── client.py
├── models.py
├── README.md
├── tests/
│   └── test_env.py          # 15 tests
└── server/
    ├── app.py
    ├── trafficops_environment.py
    ├── actions.py
    ├── grading.py            # openenv Rubric system (WeightedSum)
    ├── observations.py
    ├── tasks.py              # 5 tasks on 4x4 grid
    ├── gradio_ui.py          # Custom traffic visualization
    └── sim/
        ├── world.py
        ├── engine.py
        ├── builders.py
        └── rl_controller.py  # DQN inference (pure numpy)
```
