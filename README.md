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

## Controller Design

Intersections use **fixed-time cycling** by default — signals alternate phases at fixed intervals without responding to demand. When the LLM sets a **bias** on a direction (via `set_bias` or `set_coordination`), the controller switches to **pressure-responsive** mode, adapting phase durations based on real-time queue lengths. This means doing nothing (noop) results in substantial wasted green time, while a smart LLM agent that biases the right directions can significantly improve throughput and efficiency.

## Reward Design

Dense per-tick reward based on:
- +1.0 per cleared civilian, +5.0 per cleared emergency
- -0.002 x total waiting ticks across all vehicles
- -0.3 per wasted green tick (serving empty approaches)
- -12.0 per gridlock event

Final episode score is a weighted combination of: throughput, emergency clearance, fairness (max wait), efficiency (wasted green), planning quality (budget usage), and safety (gridlock avoidance). Weights vary by task — e.g., emergency clearance is 50% of score in the hard task.

## Setup

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

| Task | Noop | Scripted Smart | Gap |
|---|---|---|---|
| single_corridor | 0.51 | 0.67 | +0.16 |
| asymmetric_network | 0.59 | 0.70 | +0.11 |
| incident_and_emergencies | 0.36 | 0.61 | +0.25 |

Scores vary with seed (stochastic spawn timing). Noop baseline uses fixed-time cycling with no LLM intervention.

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
└── server/
    ├── app.py
    ├── trafficops_environment.py
    ├── actions.py
    ├── grading.py
    ├── observations.py
    ├── tasks.py
    └── sim/
        ├── world.py
        ├── engine.py
        └── builders.py
```
