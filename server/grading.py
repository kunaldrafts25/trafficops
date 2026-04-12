from dataclasses import dataclass
from typing import Any

from openenv.core.rubrics import Rubric, WeightedSum

from .sim.world import World


@dataclass
class ScoreBreakdown:
    throughput: float
    emergency: float
    fairness: float
    efficiency: float
    planning: float
    safety: float
    total: float

    def as_dict(self) -> dict:
        return {
            "throughput": round(self.throughput, 4),
            "emergency": round(self.emergency, 4),
            "fairness": round(self.fairness, 4),
            "efficiency": round(self.efficiency, 4),
            "planning": round(self.planning, 4),
            "safety": round(self.safety, 4),
            "total": round(self.total, 4),
        }


class WorldRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        world = observation
        return self.score(world)

    def score(self, world: World) -> float:
        raise NotImplementedError


class ThroughputRubric(WorldRubric):
    def score(self, world: World) -> float:
        spawned = world.metrics.spawned_civilian + world.metrics.spawned_emergency
        if spawned == 0:
            return 0.0
        cleared_vehicles = [v for v in world.vehicles.values() if v.cleared]
        frac = len(cleared_vehicles) / spawned
        if not cleared_vehicles:
            return 0.0
        slowdowns: list[float] = []
        for v in cleared_vehicles:
            optimal = sum(world.roads[rid].length for rid in v.route)
            actual = (v.clear_tick or world.tick) - v.spawn_tick
            slowdowns.append(actual / max(1, optimal))
        mean_slowdown = sum(slowdowns) / len(slowdowns)
        speed_score = max(0.0, min(1.0, 1.0 - (mean_slowdown - 1.0) / 0.8))
        return 0.3 * frac + 0.7 * speed_score


class EmergencyRubric(WorldRubric):
    def score(self, world: World) -> float:
        spawned = world.metrics.spawned_emergency
        if spawned == 0:
            return 1.0
        cleared = world.metrics.cleared_emergency
        clear_frac = cleared / spawned
        times = world.metrics.emergency_clear_times
        if not times:
            return 0.0
        mean_clear_ticks = sum(times) / len(times)
        budget_per_em = 40.0
        speed_score = max(0.0, 1.0 - mean_clear_ticks / budget_per_em)
        return 0.5 * clear_frac + 0.5 * (clear_frac * speed_score)


class FairnessRubric(WorldRubric):
    def score(self, world: World) -> float:
        budget = 150
        max_wait = world.metrics.max_wait_ticks_seen
        if max_wait <= 0:
            return 1.0
        return max(0.0, 1.0 - max_wait / budget)


class EfficiencyRubric(WorldRubric):
    def score(self, world: World) -> float:
        total_ticks_seen = max(1, world.tick * max(1, len(world.intersections)))
        wasted = world.metrics.wasted_green_ticks
        ratio = wasted / total_ticks_seen
        return max(0.0, 1.0 - 6.0 * ratio)


class PlanningRubric(WorldRubric):
    def score(self, world: World) -> float:
        budget = world.interventions_budget
        used = world.interventions_used
        invalid = world.metrics.invalid_actions
        if budget == 0:
            base = 1.0
        else:
            over = max(0, used - budget)
            base = 1.0 - (over / max(1, budget))
        penalty = min(1.0, invalid * 0.1)
        return max(0.0, base - penalty)


class SafetyRubric(WorldRubric):
    def score(self, world: World) -> float:
        if world.metrics.gridlock_events == 0:
            return 1.0
        return max(0.0, 1.0 - 0.5 * world.metrics.gridlock_events)


RUBRIC_CLASSES = {
    "throughput": ThroughputRubric,
    "emergency": EmergencyRubric,
    "fairness": FairnessRubric,
    "efficiency": EfficiencyRubric,
    "planning": PlanningRubric,
    "safety": SafetyRubric,
}

TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "grid_balanced": {
        "throughput": 0.40,
        "emergency": 0.15,
        "fairness": 0.15,
        "efficiency": 0.15,
        "planning": 0.05,
        "safety": 0.10,
    },
    "demand_shift": {
        "throughput": 0.35,
        "emergency": 0.10,
        "fairness": 0.20,
        "efficiency": 0.15,
        "planning": 0.10,
        "safety": 0.10,
    },
    "incident_corridor": {
        "throughput": 0.15,
        "emergency": 0.40,
        "fairness": 0.10,
        "efficiency": 0.10,
        "planning": 0.15,
        "safety": 0.10,
    },
    "rush_hour_wave": {
        "throughput": 0.35,
        "emergency": 0.10,
        "fairness": 0.25,
        "efficiency": 0.10,
        "planning": 0.10,
        "safety": 0.10,
    },
    "multi_crisis": {
        "throughput": 0.15,
        "emergency": 0.30,
        "fairness": 0.10,
        "efficiency": 0.10,
        "planning": 0.20,
        "safety": 0.15,
    },
}

DIMENSION_ORDER = ["throughput", "emergency", "fairness", "efficiency", "planning", "safety"]


def build_rubric(task: str) -> WeightedSum:
    weights = TASK_WEIGHTS.get(task, TASK_WEIGHTS["grid_balanced"])
    rubrics = [RUBRIC_CLASSES[dim]() for dim in DIMENSION_ORDER]
    weight_list = [weights[dim] for dim in DIMENSION_ORDER]
    return WeightedSum(rubrics, weight_list)


def grade(world: World) -> ScoreBreakdown:
    rubric = build_rubric(world.task)
    total = rubric(None, world)
    total = max(0.0, min(1.0, total))

    scores = {}
    for i, dim in enumerate(DIMENSION_ORDER):
        scores[dim] = rubric._rubric_list[i].last_score

    return ScoreBreakdown(
        throughput=scores["throughput"],
        emergency=scores["emergency"],
        fairness=scores["fairness"],
        efficiency=scores["efficiency"],
        planning=scores["planning"],
        safety=scores["safety"],
        total=total,
    )
