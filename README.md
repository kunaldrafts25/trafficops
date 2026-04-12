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

An OpenEnv environment where an LLM agent manages traffic signals across road networks, balancing civilian throughput, emergency vehicle prioritization, and incident response.

## Why TrafficOps?

Real-world traffic operations centers rely on human operators to coordinate signal timing, reroute traffic around incidents, and clear paths for emergency vehicles. TrafficOps simulates this workflow as a sequential decision-making problem that tests an LLM's ability to reason about spatial networks, prioritize competing objectives, and plan under uncertainty.

## Action Space

| Operation | Description |
|---|---|
| `noop` | Do nothing |
| `set_bias` | Increase green-time weight for a direction at intersections |
| `set_coordination` | Synchronize signals along a corridor for a "green wave" |
| `preempt` | Force green for a direction (emergency clearance) |
| `reroute` | Redirect vehicles around a blocked road |
| `set_policy` | Apply a policy (e.g., school_zone) to reduce phase duration |
| `cancel` | Cancel an active plan |

All actions are structured as `{op, targets, params, reason}` via `TrafficOpsAction`.

## Observation Space

Each step returns `TrafficOpsObservation` with:
- **summary** — human-readable status line with detour hints and emergency routes
- **roads** — full network topology: road IDs, from/to nodes, lengths, occupancy, queue at stop line, blocked status
- **intersections** — phase, queues per direction, bias, preempt state
- **corridors** — coordination status, flow, green-wave offsets
- **incidents** — active road blockages
- **emergencies** — live emergency vehicles with remaining route, ETA, and destination
- **active_plans** — agent's currently active interventions
- **metrics** — cleared/spawned counts, wait times, wasted green, gridlocks

## Tasks

### 1. `single_corridor` (Easy)
Three-intersection arterial with cross traffic and one ambulance. Bias signals toward the dominant arterial direction to enable demand-responsive switching, and preempt for the ambulance. Horizon: 200 ticks, budget: 5 interventions.

### 2. `asymmetric_network` (Medium)
2x2 grid with asymmetric demand that flips hard mid-episode, plus an ambulance during the transition. The agent must detect the demand flip, rebias signals, and preempt for the emergency. Horizon: 260 ticks, budget: 6 interventions.

### 3. `incident_and_emergencies` (Hard)
2x2 grid with a long-duration road blockage plus three emergency vehicles (ambulance, fire, police) arriving at staggered times. The agent must reroute civilians around the incident, preempt signals for each emergency, and bias for throughput. Horizon: 280 ticks, budget: 8 interventions.

### 4. `rush_hour_surge` (Medium-Hard)
2x2 grid where traffic demand doubles suddenly mid-episode with no advance warning. The agent must detect the surge from rising queue lengths and adapt bias accordingly, while handling an ambulance during peak chaos. Horizon: 240 ticks, budget: 6 interventions.

### 5. `multi_incident_cascade` (Expert)
2x2 grid with two incidents activating on different roads at different times, causing cascading congestion. Two emergency vehicles need rerouting around separate blockages. Tests multi-objective reasoning, sequential rerouting, and budget management across overlapping crises. Horizon: 300 ticks, budget: 10 interventions.

## Controller Design

Intersections use **fixed-time cycling** by default — signals alternate phases at fixed intervals without responding to demand. When the LLM sets a **bias** on a direction (via `set_bias` or `set_coordination`), the controller switches to **pressure-responsive** mode, adapting phase durations based on real-time queue lengths. This means doing nothing (noop) results in substantial wasted green time, while a smart LLM agent that biases the right directions can significantly improve throughput and efficiency.

## Reward Design

Dense per-tick reward based on:
- +1.0 per cleared civilian, +5.0 per cleared emergency
- -0.002 x total waiting ticks across all vehicles
- -0.3 per wasted green tick (serving empty approaches)
- -12.0 per gridlock event

Final episode score is a weighted combination of: throughput, emergency clearance, fairness (max wait), efficiency (wasted green), planning quality (budget usage), and safety (gridlock avoidance). Weights vary by task — e.g., emergency clearance is 50% of score in the hard task.

## Quick Start — Playing in the Playground

Open the **Playground** tab and follow these steps:

### Step 1: Click **Reset** to start an episode

### Step 2: Try an action

Set the fields as follows and click **Step**:

| Field | Example Value |
|---|---|
| **Op** | `set_bias` |
| **Targets** | `I1,I2,I3` |
| **Params** | `{"direction":"W","multiplier":2.5,"duration_ticks":150}` |
| **Reason** | `heavy westbound arterial demand` |

### Step 3: Observe and repeat

The response shows the full observation — queues, signal phases, metrics. Click **Step** with Op=`noop` to advance the simulation without intervening.

### Example actions to try

| Scenario | Op | Targets | Params |
|---|---|---|---|
| Bias arterial | `set_bias` | `I1,I2,I3` | `{"direction":"W","multiplier":2.5,"duration_ticks":100}` |
| Preempt for ambulance | `preempt` | `I2` | `{"direction":"N","duration_ticks":15}` |
| Reroute around blockage | `reroute` | `R_I1_I2` | `{"blocked_road":"R_I1_I2","detour":["R_I1_I3","R_I3_I4","R_I4_I2"],"duration_ticks":200}` |
| Coordinate corridor | `set_coordination` | `corridor_east` | `{"direction":"W","target_speed":0.5,"duration_ticks":100}` |
| Do nothing | `noop` | | |

## Setup (Local Development)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t trafficops-env:latest .
docker run -p 8000:8000 trafficops-env:latest
```

## Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Baseline Scores

| Task | Difficulty | Noop |
|---|---|---|
| single_corridor | Easy | 0.51 |
| asymmetric_network | Medium | 0.59 |
| rush_hour_surge | Medium-Hard | 0.60 |
| incident_and_emergencies | Hard | 0.36 |
| multi_incident_cascade | Expert | 0.50 |

Scores vary with seed (stochastic spawn timing). Noop uses fixed-time cycling with no LLM intervention. Smart agents using bias + preempt + reroute score 0.15-0.25 higher.

## Project Structure

```
trafficops/
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── inference.py
├── client.py
├── models.py
├── __init__.py
├── README.md
├── tests/
│   └── test_env.py
└── server/
    ├── app.py
    ├── trafficops_environment.py
    ├── actions.py
    ├── grading.py          # openenv Rubric system (WeightedSum)
    ├── observations.py
    ├── tasks.py             # 5 tasks: easy → expert
    └── sim/
        ├── world.py
        ├── engine.py
        └── builders.py
```
