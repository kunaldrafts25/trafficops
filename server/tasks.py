from typing import Callable

from .sim.builders import (
    add_corridor,
    add_intersection,
    add_road,
    connect_neighbors,
    new_world,
    schedule_incident,
    spawn,
    spawn_stream,
    wire,
)
from .sim.rl_controller import get_controller
from .sim.world import World


TASK_IDS = [
    "grid_balanced",
    "demand_shift",
    "incident_corridor",
    "rush_hour_wave",
    "multi_crisis",
]

GRID_ROWS = 4
GRID_COLS = 4
ROAD_LEN = 8
SOURCE_LEN = 6
SINK_LEN = 5


def build(task: str, seed: int) -> World:
    builder = _BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"unknown task: {task}")
    return builder(seed)


def _iid(r: int, c: int) -> str:
    return f"I_{r}_{c}"


def _build_grid(
    task: str,
    seed: int,
    horizon: int,
    budget: int,
) -> World:
    w = new_world(task, horizon=horizon, seed=seed, interventions_budget=budget, controller_mode="dqn")
    w.rl_controller = get_controller()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            add_intersection(w, _iid(r, c), position=(c, r), min_phase_ticks=6, max_phase_ticks=45)

    # Horizontal roads (west→east), approach direction W
    for r in range(GRID_ROWS):
        add_road(w, f"R_src_W_{r}", f"SRC_W_{r}", _iid(r, 0), approach="W", length=SOURCE_LEN)
        for c in range(GRID_COLS - 1):
            add_road(w, f"R_h_{r}_{c}", _iid(r, c), _iid(r, c + 1), approach="W", length=ROAD_LEN)
        add_road(w, f"R_sink_E_{r}", _iid(r, GRID_COLS - 1), f"SINK_E_{r}", approach="W", length=SINK_LEN)

    # Vertical roads (south→north), approach direction S
    for c in range(GRID_COLS):
        add_road(w, f"R_src_S_{c}", f"SRC_S_{c}", _iid(0, c), approach="S", length=SOURCE_LEN)
        for r in range(GRID_ROWS - 1):
            add_road(w, f"R_v_{r}_{c}", _iid(r, c), _iid(r + 1, c), approach="S", length=ROAD_LEN)
        add_road(w, f"R_sink_N_{c}", _iid(GRID_ROWS - 1, c), f"SINK_N_{c}", approach="S", length=SINK_LEN)

    # Wire each intersection
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            inc = {}
            out = {}
            inc["W"] = f"R_src_W_{r}" if c == 0 else f"R_h_{r}_{c - 1}"
            out["E"] = f"R_sink_E_{r}" if c == GRID_COLS - 1 else f"R_h_{r}_{c}"
            inc["S"] = f"R_src_S_{c}" if r == 0 else f"R_v_{r - 1}_{c}"
            out["N"] = f"R_sink_N_{c}" if r == GRID_ROWS - 1 else f"R_v_{r}_{c}"
            wire(w, _iid(r, c), incoming=inc, outgoing=out)

    connect_neighbors(w)

    # Corridors
    for r in range(GRID_ROWS):
        add_corridor(w, f"ew_row_{r}", [_iid(r, c) for c in range(GRID_COLS)], direction="W")
    for c in range(GRID_COLS):
        add_corridor(w, f"ns_col_{c}", [_iid(r, c) for r in range(GRID_ROWS)], direction="S")

    return w


def _ew_route(row: int) -> list[str]:
    route = [f"R_src_W_{row}"]
    for c in range(GRID_COLS - 1):
        route.append(f"R_h_{row}_{c}")
    route.append(f"R_sink_E_{row}")
    return route


def _ns_route(col: int) -> list[str]:
    route = [f"R_src_S_{col}"]
    for r in range(GRID_ROWS - 1):
        route.append(f"R_v_{r}_{col}")
    route.append(f"R_sink_N_{col}")
    return route


# ── Task 1: grid_balanced (Easy) ─────────────────────────────────────────
def _build_grid_balanced(seed: int) -> World:
    w = _build_grid("grid_balanced", seed, horizon=250, budget=6)

    for r in range(GRID_ROWS):
        spawn_stream(w, 2 + r, w.horizon - 30, 7, f"EW_{r}", "civilian", _ew_route(r), jitter=0.3)
    for c in range(GRID_COLS):
        spawn_stream(w, 4 + c, w.horizon - 30, 9, f"NS_{c}", "civilian", _ns_route(c), jitter=0.3)

    # One ambulance crossing east-west through the middle
    amb_tick = 80 + int(w.rng.integers(0, 30))
    spawn(w, amb_tick, "AMB_1", "ambulance", _ew_route(1))

    return w


# ── Task 2: demand_shift (Medium) ────────────────────────────────────────
def _build_demand_shift(seed: int) -> World:
    w = _build_grid("demand_shift", seed, horizon=300, budget=6)

    flip_tick = 140 + int(w.rng.integers(0, 20))

    # Phase A: heavy north-south, light east-west
    for c in range(GRID_COLS):
        spawn_stream(w, 2 + c, flip_tick, 4, f"NS_A_{c}", "civilian", _ns_route(c), jitter=0.25)
    for r in range(GRID_ROWS):
        spawn_stream(w, 4 + r, flip_tick, 18, f"EW_A_{r}", "civilian", _ew_route(r), jitter=0.3)

    # Phase B: demand flips — heavy east-west, light north-south
    for r in range(GRID_ROWS):
        spawn_stream(w, flip_tick, w.horizon - 30, 4, f"EW_B_{r}", "civilian", _ew_route(r), jitter=0.25)
    for c in range(GRID_COLS):
        spawn_stream(w, flip_tick, w.horizon - 30, 18, f"NS_B_{c}", "civilian", _ns_route(c), jitter=0.3)

    # Ambulance during transition
    amb_tick = flip_tick + int(w.rng.integers(5, 20))
    spawn(w, amb_tick, "AMB_1", "ambulance", _ns_route(2))

    return w


# ── Task 3: incident_corridor (Hard) ─────────────────────────────────────
def _build_incident_corridor(seed: int) -> World:
    w = _build_grid("incident_corridor", seed, horizon=280, budget=8)

    for r in range(GRID_ROWS):
        spawn_stream(w, 2 + r, w.horizon - 30, 8, f"EW_{r}", "civilian", _ew_route(r), jitter=0.3)
    for c in range(GRID_COLS):
        spawn_stream(w, 4 + c, w.horizon - 30, 9, f"NS_{c}", "civilian", _ns_route(c), jitter=0.3)

    # Incident blocks row 1 mid-corridor
    inc_tick = 50 + int(w.rng.integers(0, 15))
    inc_end = inc_tick + 150 + int(w.rng.integers(0, 30))
    schedule_incident(w, inc_tick, "INC_1", "R_h_1_1", "accident", inc_end)

    # Ambulance whose route goes through blocked road
    amb_tick = inc_tick + 20 + int(w.rng.integers(0, 10))
    spawn(w, amb_tick, "AMB_1", "ambulance", _ew_route(1))

    # Fire truck on north-south (doesn't hit incident but needs preempt)
    fire_tick = inc_tick + 60 + int(w.rng.integers(0, 20))
    spawn(w, fire_tick, "FIRE_1", "fire", _ns_route(3))

    return w


# ── Task 4: rush_hour_wave (Hard) ────────────────────────────────────────
def _build_rush_hour_wave(seed: int) -> World:
    w = _build_grid("rush_hour_wave", seed, horizon=280, budget=8)

    surge_tick = 90 + int(w.rng.integers(0, 20))

    # Phase 1: light balanced traffic
    for r in range(GRID_ROWS):
        spawn_stream(w, 2 + r, surge_tick, 14, f"EW_L_{r}", "civilian", _ew_route(r), jitter=0.3)
    for c in range(GRID_COLS):
        spawn_stream(w, 4 + c, surge_tick, 14, f"NS_L_{c}", "civilian", _ns_route(c), jitter=0.3)

    # Phase 2: demand triples from the south — wave ripples north
    for c in range(GRID_COLS):
        spawn_stream(w, surge_tick, w.horizon - 30, 3, f"NS_H_{c}", "civilian", _ns_route(c), jitter=0.25)
    # East-west stays moderate
    for r in range(GRID_ROWS):
        spawn_stream(w, surge_tick, w.horizon - 30, 10, f"EW_H_{r}", "civilian", _ew_route(r), jitter=0.3)

    # Police car during peak
    pol_tick = surge_tick + 30 + int(w.rng.integers(0, 20))
    spawn(w, pol_tick, "POLICE_1", "police", _ew_route(2))

    return w


# ── Task 5: multi_crisis (Expert) ────────────────────────────────────────
def _build_multi_crisis(seed: int) -> World:
    w = _build_grid("multi_crisis", seed, horizon=320, budget=12)

    # Moderate asymmetric traffic (heavier east-west)
    for r in range(GRID_ROWS):
        spawn_stream(w, 2 + r, w.horizon - 40, 6, f"EW_{r}", "civilian", _ew_route(r), jitter=0.3)
    for c in range(GRID_COLS):
        spawn_stream(w, 4 + c, w.horizon - 40, 10, f"NS_{c}", "civilian", _ns_route(c), jitter=0.3)

    # Incident 1: blocks row 0 at tick ~50
    inc1_tick = 45 + int(w.rng.integers(0, 15))
    inc1_end = inc1_tick + 120 + int(w.rng.integers(0, 30))
    schedule_incident(w, inc1_tick, "INC_1", "R_h_0_1", "accident", inc1_end)

    # Incident 2: blocks column 2 at tick ~130
    inc2_tick = 120 + int(w.rng.integers(0, 20))
    inc2_end = inc2_tick + 100 + int(w.rng.integers(0, 20))
    schedule_incident(w, inc2_tick, "INC_2", "R_v_2_2", "construction", inc2_end)

    # Emergency 1: ambulance through incident 1 zone
    amb_tick = inc1_tick + 20 + int(w.rng.integers(0, 10))
    spawn(w, amb_tick, "AMB_1", "ambulance", _ew_route(0))

    # Emergency 2: fire truck through incident 2 zone
    fire_tick = inc2_tick + 15 + int(w.rng.integers(0, 10))
    spawn(w, fire_tick, "FIRE_1", "fire", _ns_route(2))

    # Emergency 3: police car diagonal route (late)
    pol_tick = fire_tick + 40 + int(w.rng.integers(0, 15))
    spawn(w, pol_tick, "POLICE_1", "police", _ew_route(3))

    return w


_BUILDERS: dict[str, Callable[[int], World]] = {
    "grid_balanced": _build_grid_balanced,
    "demand_shift": _build_demand_shift,
    "incident_corridor": _build_incident_corridor,
    "rush_hour_wave": _build_rush_hour_wave,
    "multi_crisis": _build_multi_crisis,
}
