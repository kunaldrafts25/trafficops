from typing import Optional

import numpy as np

from .world import (
    Corridor,
    Direction,
    Incident,
    IncidentEvent,
    IncidentKind,
    Intersection,
    Metrics,
    Road,
    SpawnEvent,
    VehicleType,
    World,
)


PHASES_NS_EW: list[frozenset[Direction]] = [
    frozenset({"N", "S"}),
    frozenset({"E", "W"}),
]


def new_world(
    task: str,
    horizon: int,
    seed: int,
    interventions_budget: int,
    controller_mode: str = "max_pressure",
) -> World:
    return World(
        tick=0,
        horizon=horizon,
        task=task,
        seed=seed,
        rng=np.random.default_rng(seed),
        roads={},
        intersections={},
        corridors={},
        metrics=Metrics(),
        interventions_budget=interventions_budget,
        controller_mode=controller_mode,
    )


def add_intersection(
    world: World,
    iid: str,
    position: tuple[int, int],
    min_phase_ticks: int = 6,
    max_phase_ticks: int = 45,
    phases: Optional[list[frozenset[Direction]]] = None,
) -> Intersection:
    I = Intersection(
        id=iid,
        position=position,
        phases=phases or PHASES_NS_EW,
        min_phase_ticks=min_phase_ticks,
        max_phase_ticks=max_phase_ticks,
    )
    world.intersections[iid] = I
    return I


def add_road(
    world: World,
    rid: str,
    from_node: str,
    to_node: str,
    approach: Direction,
    length: int,
) -> Road:
    road = Road(
        id=rid,
        from_node=from_node,
        to_node=to_node,
        approach_direction=approach,
        length=length,
    )
    world.roads[rid] = road
    return road


def wire(
    world: World,
    iid: str,
    incoming: dict[Direction, str],
    outgoing: dict[Direction, str],
) -> None:
    I = world.intersections[iid]
    I.incoming = dict(incoming)
    I.outgoing = dict(outgoing)


def connect_neighbors(world: World) -> None:
    for I in world.intersections.values():
        neighbors: list[str] = []
        for rid in I.incoming.values():
            fn = world.roads[rid].from_node
            if fn in world.intersections and fn not in neighbors:
                neighbors.append(fn)
        for rid in I.outgoing.values():
            tn = world.roads[rid].to_node
            if tn in world.intersections and tn not in neighbors:
                neighbors.append(tn)
        I.neighbors = neighbors


def add_corridor(
    world: World,
    cid: str,
    intersections: list[str],
    direction: Direction,
) -> Corridor:
    c = Corridor(id=cid, intersections=list(intersections), direction=direction)
    world.corridors[cid] = c
    return c


def spawn(
    world: World,
    at_tick: int,
    vid: str,
    vtype: VehicleType,
    route: list[str],
) -> None:
    world.spawn_schedule.append(
        SpawnEvent(tick=at_tick, vehicle_id=vid, vehicle_type=vtype, route=list(route))
    )


def spawn_stream(
    world: World,
    start_tick: int,
    end_tick: int,
    period: int,
    vid_prefix: str,
    vtype: VehicleType,
    route: list[str],
    jitter: float = 0.0,
) -> int:
    n = 0
    t = start_tick
    while t < end_tick:
        offset = 0
        if jitter > 0:
            offset = int(world.rng.integers(-int(period * jitter), int(period * jitter) + 1))
        actual_tick = max(1, t + offset)
        spawn(world, actual_tick, f"{vid_prefix}_{n}", vtype, route)
        n += 1
        t += period
    return n


def schedule_incident(
    world: World,
    at_tick: int,
    incident_id: str,
    road_id: str,
    kind: IncidentKind,
    end_tick: Optional[int] = None,
) -> None:
    inc = Incident(
        id=incident_id,
        road_id=road_id,
        kind=kind,
        start_tick=at_tick,
        end_tick=end_tick,
    )
    world.incident_schedule.append(IncidentEvent(tick=at_tick, incident=inc))
