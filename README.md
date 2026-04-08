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
- **summary** — human-readable status line
- **intersections** — phase, queues per direction, bias, preempt state
- **corridors** — coordination status, flow
- **incidents** — active road blockages
- **emergencies** — live emergency vehicles with ETA
- **active_plans** — agent's currently active interventions
- **metrics** — cleared/spawned counts, wait times, wasted green, gridlocks

## Tasks

### 1. `single_corridor` (Easy)
Three-intersection arterial with cross traffic. Optimize signal timing to clear a steady stream of vehicles. Horizon: 160 ticks, budget: 5 interventions.

### 2. `asymmetric_network` (Medium)
2x2 grid with asymmetric demand that shifts mid-episode. The agent must detect the demand flip and reallocate green time. Horizon: 240 ticks, budget: 6 interventions.

### 3. `incident_and_emergencies` (Hard)
2x2 grid with a long-duration road blockage plus three emergency vehicles (ambulance, fire, police) arriving at staggered times. The agent must reroute civilians, preempt signals for emergencies, and maintain throughput. Horizon: 280 ticks, budget: 8 interventions.

## Reward Design

Dense per-tick reward based on:
- +1.0 per cleared civilian, +5.0 per cleared emergency
- -0.002 × total waiting ticks across all vehicles
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
docker build -t trafficops-env:latest -f server/Dockerfile .
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

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score |
|---|---|
| single_corridor | ~0.55 |
| asymmetric_network | ~0.45 |
| incident_and_emergencies | ~0.35 |

## Project Structure

```
trafficops/
├── openenv.yaml
├── pyproject.toml
├── inference.py
├── models.py
├── __init__.py
├── README.md
└── server/
    ├── Dockerfile
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
