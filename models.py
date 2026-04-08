from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ActionOp = Literal[
    "noop",
    "set_coordination",
    "set_bias",
    "preempt",
    "reroute",
    "set_policy",
    "cancel",
]

Direction = Literal["N", "S", "E", "W"]
TaskId = Literal["single_corridor", "asymmetric_network", "incident_and_emergencies"]


class TrafficOpsAction(Action):
    op: ActionOp = "noop"
    targets: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    plan_id: Optional[str] = None
    reason: str = ""


class IntersectionView(BaseModel):
    id: str
    position: tuple[int, int]
    current_phase: str
    phase_timer: int
    min_phase_ticks: int
    max_phase_ticks: int
    queues: dict[Direction, int]
    bias: dict[Direction, float]
    preempt_direction: Optional[Direction] = None
    preempt_expires_tick: Optional[int] = None
    neighbors: list[str] = Field(default_factory=list)


class CorridorView(BaseModel):
    id: str
    intersections: list[str]
    direction: Direction
    coordinated: bool
    plan_id: Optional[str] = None
    target_speed: Optional[float] = None
    total_flow: int


class IncidentView(BaseModel):
    id: str
    road_id: str
    kind: Literal["accident", "construction", "debris"]
    start_tick: int
    end_tick: Optional[int]
    active: bool


class EmergencyView(BaseModel):
    id: str
    type: Literal["ambulance", "fire", "police"]
    origin: str
    destination: str
    current_road: Optional[str]
    ticks_since_spawn: int
    eta_ticks: Optional[int]
    cleared: bool


class PlanView(BaseModel):
    id: str
    op: ActionOp
    created_tick: int
    expires_tick: Optional[int]
    targets: list[str]
    params: dict[str, Any]


class MetricsView(BaseModel):
    cleared_civilian: int
    cleared_emergency: int
    spawned_civilian: int
    spawned_emergency: int
    mean_wait_ticks: float
    max_wait_ticks: int
    total_queue: int
    wasted_green_ticks: int
    gridlock_events: int
    conflicting_plans: int


class TrafficOpsObservation(Observation):
    task: TaskId = "single_corridor"
    tick: int = 0
    horizon: int = 0
    summary: str = ""
    intersections: list[IntersectionView] = Field(default_factory=list)
    corridors: list[CorridorView] = Field(default_factory=list)
    incidents: list[IncidentView] = Field(default_factory=list)
    emergencies: list[EmergencyView] = Field(default_factory=list)
    active_plans: list[PlanView] = Field(default_factory=list)
    metrics: Optional[MetricsView] = None
    interventions_used: int = 0
    interventions_budget: int = 0
    last_action_error: Optional[str] = None


class TrafficOpsState(State):
    task: TaskId = "single_corridor"
    tick: int = 0
    horizon: int = 0
    seed: int = 0
    final_score: Optional[float] = None
