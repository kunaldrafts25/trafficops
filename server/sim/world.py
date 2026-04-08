from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


Direction = Literal["N", "S", "E", "W"]
VehicleType = Literal["civilian", "ambulance", "fire", "police", "bus"]
IncidentKind = Literal["accident", "construction", "debris"]

EMERGENCY_TYPES: set[str] = {"ambulance", "fire", "police"}
OPPOSITE: dict[Direction, Direction] = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class Road:
    id: str
    from_node: str
    to_node: str
    approach_direction: Direction
    length: int
    cells: list[Optional[str]] = field(default_factory=list)
    blocked: bool = False

    def __post_init__(self):
        if not self.cells:
            self.cells = [None] * self.length

    def occupancy(self) -> int:
        return sum(1 for c in self.cells if c is not None)

    def queue_at_tail(self) -> int:
        n = 0
        for i in range(self.length - 1, -1, -1):
            if self.cells[i] is not None:
                n += 1
            else:
                break
        return n


@dataclass
class Intersection:
    id: str
    position: tuple[int, int]
    phases: list[frozenset[Direction]]
    current_phase_idx: int = 0
    phase_timer: int = 0
    min_phase_ticks: int = 6
    max_phase_ticks: int = 45
    incoming: dict[Direction, str] = field(default_factory=dict)
    outgoing: dict[Direction, str] = field(default_factory=dict)
    bias: dict[Direction, float] = field(default_factory=lambda: {"N": 1.0, "S": 1.0, "E": 1.0, "W": 1.0})
    preempt_direction: Optional[Direction] = None
    preempt_expires_tick: Optional[int] = None
    neighbors: list[str] = field(default_factory=list)

    def current_phase(self) -> frozenset[Direction]:
        return self.phases[self.current_phase_idx]

    def phase_name(self) -> str:
        return "+".join(sorted(self.current_phase()))

    def phase_idx_containing(self, direction: Direction) -> Optional[int]:
        for i, ph in enumerate(self.phases):
            if direction in ph:
                return i
        return None


@dataclass
class Vehicle:
    id: str
    type: VehicleType
    route: list[str]
    route_idx: int = 0
    position_in_road: int = 0
    spawn_tick: int = 0
    wait_ticks: int = 0
    cleared: bool = False
    clear_tick: Optional[int] = None

    def is_emergency(self) -> bool:
        return self.type in EMERGENCY_TYPES


@dataclass
class Incident:
    id: str
    road_id: str
    kind: IncidentKind
    start_tick: int
    end_tick: Optional[int]
    active: bool = False
    described: bool = False


@dataclass
class Plan:
    id: str
    op: str
    created_tick: int
    expires_tick: Optional[int]
    targets: list[str]
    params: dict
    reason: str = ""
    snapshot: dict = field(default_factory=dict)


@dataclass
class Corridor:
    id: str
    intersections: list[str]
    direction: Direction
    coordinated: bool = False
    plan_id: Optional[str] = None
    target_speed: Optional[float] = None


@dataclass
class SpawnEvent:
    tick: int
    vehicle_id: str
    vehicle_type: VehicleType
    route: list[str]


@dataclass
class IncidentEvent:
    tick: int
    incident: Incident


@dataclass
class Metrics:
    cleared_civilian: int = 0
    cleared_emergency: int = 0
    spawned_civilian: int = 0
    spawned_emergency: int = 0
    wasted_green_ticks: int = 0
    gridlock_events: int = 0
    emergency_clear_times: list[int] = field(default_factory=list)
    max_wait_ticks_seen: int = 0
    invalid_actions: int = 0
    stalled_streak: int = 0


@dataclass
class TickStats:
    cleared_civ: int = 0
    cleared_em: int = 0
    wasted_green: int = 0
    gridlock: int = 0
    moved_any: bool = False


@dataclass
class World:
    tick: int
    horizon: int
    task: str
    seed: int
    rng: np.random.Generator
    roads: dict[str, Road]
    intersections: dict[str, Intersection]
    corridors: dict[str, Corridor]
    vehicles: dict[str, Vehicle] = field(default_factory=dict)
    incidents: list[Incident] = field(default_factory=list)
    active_plans: dict[str, Plan] = field(default_factory=dict)
    spawn_schedule: list[SpawnEvent] = field(default_factory=list)
    incident_schedule: list[IncidentEvent] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    event_log: list[str] = field(default_factory=list)
    interventions_used: int = 0
    interventions_budget: int = 0
    last_action_error: Optional[str] = None
    next_plan_seq: int = 0
    reroute_overrides: dict[str, list[str]] = field(default_factory=dict)

    def log(self, msg: str) -> None:
        self.event_log.append(f"t={self.tick} {msg}")
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def new_plan_id(self) -> str:
        self.next_plan_seq += 1
        return f"plan_{self.next_plan_seq:04d}"
