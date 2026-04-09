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
from .sim.world import World


TASK_IDS = [
    "single_corridor",
    "asymmetric_network",
    "incident_and_emergencies",
    "rush_hour_surge",
    "multi_incident_cascade",
]


def build(task: str, seed: int) -> World:
    builder = _BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"unknown task: {task}")
    return builder(seed)


def _build_single_corridor(seed: int) -> World:
    horizon = 200
    w = new_world("single_corridor", horizon=horizon, seed=seed, interventions_budget=5)

    for i, x in enumerate([0, 1, 2], start=1):
        add_intersection(w, f"I{i}", position=(x, 0), min_phase_ticks=8, max_phase_ticks=22)

    # Long inter-intersection roads → platoon coordination matters
    arterial_len = 10
    source_len = 6
    sink_len = 5

    add_road(w, "R_W_I1", "SRC_W", "I1", approach="W", length=source_len)
    add_road(w, "R_I1_I2", "I1", "I2", approach="W", length=arterial_len)
    add_road(w, "R_I2_I3", "I2", "I3", approach="W", length=arterial_len)
    add_road(w, "R_I3_E", "I3", "SINK_E", approach="W", length=sink_len)

    for iid in ("I1", "I2", "I3"):
        add_road(w, f"R_N_{iid}", f"SRC_N_{iid}", iid, approach="N", length=source_len)
        add_road(w, f"R_{iid}_S", iid, f"SINK_S_{iid}", approach="N", length=sink_len)

    wire(w, "I1",
         incoming={"W": "R_W_I1", "N": "R_N_I1"},
         outgoing={"E": "R_I1_I2", "S": "R_I1_S"})
    wire(w, "I2",
         incoming={"W": "R_I1_I2", "N": "R_N_I2"},
         outgoing={"E": "R_I2_I3", "S": "R_I2_S"})
    wire(w, "I3",
         incoming={"W": "R_I2_I3", "N": "R_N_I3"},
         outgoing={"E": "R_I3_E", "S": "R_I3_S"})

    connect_neighbors(w)
    add_corridor(w, "corridor_east", intersections=["I1", "I2", "I3"], direction="W")

    arterial_route = ["R_W_I1", "R_I1_I2", "R_I2_I3", "R_I3_E"]
    spawn_stream(w, start_tick=2, end_tick=horizon - 20, period=3,
                 vid_prefix="ART", vtype="civilian", route=arterial_route, jitter=0.3)

    for iid in ("I1", "I2", "I3"):
        route = [f"R_N_{iid}", f"R_{iid}_S"]
        spawn_stream(w, start_tick=5, end_tick=horizon - 20, period=10,
                     vid_prefix=f"CROSS_{iid}", vtype="civilian", route=route, jitter=0.3)

    # Emergency crosses arterial — stochastic timing
    amb_tick = 70 + int(w.rng.integers(0, 30))
    spawn(w, at_tick=amb_tick, vid="AMB_1", vtype="ambulance",
          route=["R_N_I2", "R_I2_S"])

    return w


def _build_asymmetric_network(seed: int) -> World:
    horizon = 260
    w = new_world("asymmetric_network", horizon=horizon, seed=seed, interventions_budget=6)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=8, max_phase_ticks=22)

    seg = 7
    src = 6
    snk = 5

    add_road(w, "R_SW_I3", "SRC_SW", "I3", approach="S", length=src)
    add_road(w, "R_SE_I4", "SRC_SE", "I4", approach="S", length=src)
    add_road(w, "R_I3_I1", "I3", "I1", approach="S", length=seg)
    add_road(w, "R_I4_I2", "I4", "I2", approach="S", length=seg)
    add_road(w, "R_I1_N", "I1", "SINK_N1", approach="S", length=snk)
    add_road(w, "R_I2_N", "I2", "SINK_N2", approach="S", length=snk)

    add_road(w, "R_WW_I1", "SRC_WW1", "I1", approach="W", length=src)
    add_road(w, "R_WW_I3", "SRC_WW3", "I3", approach="W", length=src)
    add_road(w, "R_I1_I2", "I1", "I2", approach="W", length=seg)
    add_road(w, "R_I3_I4", "I3", "I4", approach="W", length=seg)
    add_road(w, "R_I2_E", "I2", "SINK_E2", approach="W", length=snk)
    add_road(w, "R_I4_E", "I4", "SINK_E4", approach="W", length=snk)

    wire(w, "I3",
         incoming={"S": "R_SW_I3", "W": "R_WW_I3"},
         outgoing={"N": "R_I3_I1", "E": "R_I3_I4"})
    wire(w, "I4",
         incoming={"S": "R_SE_I4", "W": "R_I3_I4"},
         outgoing={"N": "R_I4_I2", "E": "R_I4_E"})
    wire(w, "I1",
         incoming={"S": "R_I3_I1", "W": "R_WW_I1"},
         outgoing={"N": "R_I1_N", "E": "R_I1_I2"})
    wire(w, "I2",
         incoming={"S": "R_I4_I2", "W": "R_I1_I2"},
         outgoing={"N": "R_I2_N", "E": "R_I2_E"})

    connect_neighbors(w)
    add_corridor(w, "arterial_north_west", intersections=["I3", "I1"], direction="S")
    add_corridor(w, "arterial_north_east", intersections=["I4", "I2"], direction="S")
    add_corridor(w, "arterial_east_north", intersections=["I1", "I2"], direction="W")
    add_corridor(w, "arterial_east_south", intersections=["I3", "I4"], direction="W")

    phase_a_end = 140

    spawn_stream(w, 2, phase_a_end, 2, "NA_W", "civilian", ["R_SW_I3", "R_I3_I1", "R_I1_N"], jitter=0.25)
    spawn_stream(w, 3, phase_a_end, 2, "NA_E", "civilian", ["R_SE_I4", "R_I4_I2", "R_I2_N"], jitter=0.25)
    spawn_stream(w, 6, phase_a_end, 30, "XA_N", "civilian", ["R_WW_I1", "R_I1_I2", "R_I2_E"], jitter=0.3)
    spawn_stream(w, 8, phase_a_end, 30, "XA_S", "civilian", ["R_WW_I3", "R_I3_I4", "R_I4_E"], jitter=0.3)

    spawn_stream(w, phase_a_end, horizon - 20, 30, "NB_W", "civilian", ["R_SW_I3", "R_I3_I1", "R_I1_N"], jitter=0.3)
    spawn_stream(w, phase_a_end + 1, horizon - 20, 30, "NB_E", "civilian", ["R_SE_I4", "R_I4_I2", "R_I2_N"], jitter=0.3)
    spawn_stream(w, phase_a_end + 2, horizon - 20, 2, "EB_N", "civilian", ["R_WW_I1", "R_I1_I2", "R_I2_E"], jitter=0.25)
    spawn_stream(w, phase_a_end + 3, horizon - 20, 2, "EB_S", "civilian", ["R_WW_I3", "R_I3_I4", "R_I4_E"], jitter=0.25)

    # Emergency during demand flip — stochastic timing
    amb_tick = phase_a_end + int(w.rng.integers(5, 25))
    spawn(w, at_tick=amb_tick, vid="AMB_1", vtype="ambulance",
          route=["R_SW_I3", "R_I3_I1", "R_I1_N"])

    return w


def _build_incident_and_emergencies(seed: int) -> World:
    horizon = 280
    w = new_world("incident_and_emergencies", horizon=horizon, seed=seed, interventions_budget=8)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=8, max_phase_ticks=22)

    seg = 7
    src = 6
    snk = 5

    add_road(w, "R_S_I3", "SRC_S3", "I3", approach="S", length=src)
    add_road(w, "R_S_I4", "SRC_S4", "I4", approach="S", length=src)
    add_road(w, "R_I3_I1", "I3", "I1", approach="S", length=seg)
    add_road(w, "R_I4_I2", "I4", "I2", approach="S", length=seg)
    add_road(w, "R_I1_N", "I1", "SINK_N1", approach="S", length=snk)
    add_road(w, "R_I2_N", "I2", "SINK_N2", approach="S", length=snk)

    add_road(w, "R_W_I1", "SRC_W1", "I1", approach="W", length=src)
    add_road(w, "R_W_I3", "SRC_W3", "I3", approach="W", length=src)
    add_road(w, "R_I1_I2", "I1", "I2", approach="W", length=seg)
    add_road(w, "R_I3_I4", "I3", "I4", approach="W", length=seg)
    add_road(w, "R_I2_E", "I2", "SINK_E2", approach="W", length=snk)
    add_road(w, "R_I4_E", "I4", "SINK_E4", approach="W", length=snk)

    add_road(w, "R_I1_I3", "I1", "I3", approach="N", length=seg)
    add_road(w, "R_I2_I4", "I2", "I4", approach="N", length=seg)

    wire(w, "I3",
         incoming={"S": "R_S_I3", "W": "R_W_I3", "N": "R_I1_I3"},
         outgoing={"N": "R_I3_I1", "E": "R_I3_I4"})
    wire(w, "I4",
         incoming={"S": "R_S_I4", "W": "R_I3_I4", "N": "R_I2_I4"},
         outgoing={"N": "R_I4_I2", "E": "R_I4_E"})
    wire(w, "I1",
         incoming={"S": "R_I3_I1", "W": "R_W_I1"},
         outgoing={"N": "R_I1_N", "E": "R_I1_I2", "S": "R_I1_I3"})
    wire(w, "I2",
         incoming={"S": "R_I4_I2", "W": "R_I1_I2"},
         outgoing={"N": "R_I2_N", "E": "R_I2_E", "S": "R_I2_I4"})

    connect_neighbors(w)
    add_corridor(w, "arterial_north_west", intersections=["I3", "I1"], direction="S")
    add_corridor(w, "arterial_north_east", intersections=["I4", "I2"], direction="S")
    add_corridor(w, "arterial_east_north", intersections=["I1", "I2"], direction="W")
    add_corridor(w, "arterial_east_south", intersections=["I3", "I4"], direction="W")

    spawn_stream(w, 2, horizon - 40, 8, "CIV_N1", "civilian", ["R_S_I3", "R_I3_I1", "R_I1_N"], jitter=0.3)
    spawn_stream(w, 4, horizon - 40, 9, "CIV_N2", "civilian", ["R_S_I4", "R_I4_I2", "R_I2_N"], jitter=0.3)
    spawn_stream(w, 6, horizon - 40, 9, "CIV_E1", "civilian", ["R_W_I1", "R_I1_I2", "R_I2_E"], jitter=0.3)
    spawn_stream(w, 8, horizon - 40, 10, "CIV_E2", "civilian", ["R_W_I3", "R_I3_I4", "R_I4_E"], jitter=0.3)

    # Stochastic incident timing
    inc_tick = 50 + int(w.rng.integers(0, 20))
    inc_end = inc_tick + 150 + int(w.rng.integers(0, 40))
    schedule_incident(w, at_tick=inc_tick, incident_id="INC_1",
                      road_id="R_I1_I2", kind="accident", end_tick=inc_end)

    # Staggered emergencies — routes through blocked area need reroute
    amb_tick = inc_tick + 25 + int(w.rng.integers(0, 15))
    spawn(w, at_tick=amb_tick, vid="AMB_1", vtype="ambulance",
          route=["R_S_I3", "R_I3_I1", "R_I1_I2", "R_I2_E"])

    fire_tick = amb_tick + 35 + int(w.rng.integers(0, 20))
    spawn(w, at_tick=fire_tick, vid="FIRE_1", vtype="fire",
          route=["R_S_I4", "R_I4_I2", "R_I2_N"])

    police_tick = fire_tick + 30 + int(w.rng.integers(0, 15))
    spawn(w, at_tick=police_tick, vid="POLICE_1", vtype="police",
          route=["R_W_I1", "R_I1_I2", "R_I2_E"])

    return w


def _build_rush_hour_surge(seed: int) -> World:
    horizon = 240
    w = new_world("rush_hour_surge", horizon=horizon, seed=seed, interventions_budget=6)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=8, max_phase_ticks=22)

    seg = 6
    src = 5
    snk = 4

    add_road(w, "R_S_I3", "SRC_S3", "I3", approach="S", length=src)
    add_road(w, "R_S_I4", "SRC_S4", "I4", approach="S", length=src)
    add_road(w, "R_I3_I1", "I3", "I1", approach="S", length=seg)
    add_road(w, "R_I4_I2", "I4", "I2", approach="S", length=seg)
    add_road(w, "R_I1_N", "I1", "SINK_N1", approach="S", length=snk)
    add_road(w, "R_I2_N", "I2", "SINK_N2", approach="S", length=snk)

    add_road(w, "R_W_I1", "SRC_W1", "I1", approach="W", length=src)
    add_road(w, "R_W_I3", "SRC_W3", "I3", approach="W", length=src)
    add_road(w, "R_I1_I2", "I1", "I2", approach="W", length=seg)
    add_road(w, "R_I3_I4", "I3", "I4", approach="W", length=seg)
    add_road(w, "R_I2_E", "I2", "SINK_E2", approach="W", length=snk)
    add_road(w, "R_I4_E", "I4", "SINK_E4", approach="W", length=snk)

    wire(w, "I3",
         incoming={"S": "R_S_I3", "W": "R_W_I3"},
         outgoing={"N": "R_I3_I1", "E": "R_I3_I4"})
    wire(w, "I4",
         incoming={"S": "R_S_I4", "W": "R_I3_I4"},
         outgoing={"N": "R_I4_I2", "E": "R_I4_E"})
    wire(w, "I1",
         incoming={"S": "R_I3_I1", "W": "R_W_I1"},
         outgoing={"N": "R_I1_N", "E": "R_I1_I2"})
    wire(w, "I2",
         incoming={"S": "R_I4_I2", "W": "R_I1_I2"},
         outgoing={"N": "R_I2_N", "E": "R_I2_E"})

    connect_neighbors(w)
    add_corridor(w, "arterial_north", intersections=["I3", "I1"], direction="S")
    add_corridor(w, "arterial_east", intersections=["I1", "I2"], direction="W")

    surge_tick = 100 + int(w.rng.integers(0, 20))

    # Phase 1: light balanced traffic
    spawn_stream(w, 2, surge_tick, 12, "LN1", "civilian", ["R_S_I3", "R_I3_I1", "R_I1_N"], jitter=0.3)
    spawn_stream(w, 4, surge_tick, 12, "LN2", "civilian", ["R_S_I4", "R_I4_I2", "R_I2_N"], jitter=0.3)
    spawn_stream(w, 6, surge_tick, 14, "LE1", "civilian", ["R_W_I1", "R_I1_I2", "R_I2_E"], jitter=0.3)
    spawn_stream(w, 8, surge_tick, 14, "LE2", "civilian", ["R_W_I3", "R_I3_I4", "R_I4_E"], jitter=0.3)

    # Phase 2: demand doubles suddenly on ALL directions — rush hour hits
    spawn_stream(w, surge_tick, horizon - 20, 4, "HN1", "civilian", ["R_S_I3", "R_I3_I1", "R_I1_N"], jitter=0.3)
    spawn_stream(w, surge_tick, horizon - 20, 4, "HN2", "civilian", ["R_S_I4", "R_I4_I2", "R_I2_N"], jitter=0.3)
    spawn_stream(w, surge_tick, horizon - 20, 5, "HE1", "civilian", ["R_W_I1", "R_I1_I2", "R_I2_E"], jitter=0.3)
    spawn_stream(w, surge_tick, horizon - 20, 5, "HE2", "civilian", ["R_W_I3", "R_I3_I4", "R_I4_E"], jitter=0.3)

    # Emergency during peak chaos
    amb_tick = surge_tick + 20 + int(w.rng.integers(0, 15))
    spawn(w, at_tick=amb_tick, vid="AMB_1", vtype="ambulance",
          route=["R_W_I1", "R_I1_I2", "R_I2_E"])

    return w


def _build_multi_incident_cascade(seed: int) -> World:
    horizon = 300
    w = new_world("multi_incident_cascade", horizon=horizon, seed=seed, interventions_budget=10)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=8, max_phase_ticks=22)

    seg = 7
    src = 6
    snk = 5

    add_road(w, "R_S_I3", "SRC_S3", "I3", approach="S", length=src)
    add_road(w, "R_S_I4", "SRC_S4", "I4", approach="S", length=src)
    add_road(w, "R_I3_I1", "I3", "I1", approach="S", length=seg)
    add_road(w, "R_I4_I2", "I4", "I2", approach="S", length=seg)
    add_road(w, "R_I1_N", "I1", "SINK_N1", approach="S", length=snk)
    add_road(w, "R_I2_N", "I2", "SINK_N2", approach="S", length=snk)

    add_road(w, "R_W_I1", "SRC_W1", "I1", approach="W", length=src)
    add_road(w, "R_W_I3", "SRC_W3", "I3", approach="W", length=src)
    add_road(w, "R_I1_I2", "I1", "I2", approach="W", length=seg)
    add_road(w, "R_I3_I4", "I3", "I4", approach="W", length=seg)
    add_road(w, "R_I2_E", "I2", "SINK_E2", approach="W", length=snk)
    add_road(w, "R_I4_E", "I4", "SINK_E4", approach="W", length=snk)

    # Bidirectional links for detour routes
    add_road(w, "R_I1_I3", "I1", "I3", approach="N", length=seg)
    add_road(w, "R_I2_I4", "I2", "I4", approach="N", length=seg)

    wire(w, "I3",
         incoming={"S": "R_S_I3", "W": "R_W_I3", "N": "R_I1_I3"},
         outgoing={"N": "R_I3_I1", "E": "R_I3_I4"})
    wire(w, "I4",
         incoming={"S": "R_S_I4", "W": "R_I3_I4", "N": "R_I2_I4"},
         outgoing={"N": "R_I4_I2", "E": "R_I4_E"})
    wire(w, "I1",
         incoming={"S": "R_I3_I1", "W": "R_W_I1"},
         outgoing={"N": "R_I1_N", "E": "R_I1_I2", "S": "R_I1_I3"})
    wire(w, "I2",
         incoming={"S": "R_I4_I2", "W": "R_I1_I2"},
         outgoing={"N": "R_I2_N", "E": "R_I2_E", "S": "R_I2_I4"})

    connect_neighbors(w)
    add_corridor(w, "arterial_north", intersections=["I3", "I1"], direction="S")
    add_corridor(w, "arterial_east", intersections=["I3", "I4"], direction="W")

    # Steady traffic from all directions
    spawn_stream(w, 2, horizon - 40, 7, "CIV_N1", "civilian", ["R_S_I3", "R_I3_I1", "R_I1_N"], jitter=0.3)
    spawn_stream(w, 4, horizon - 40, 8, "CIV_N2", "civilian", ["R_S_I4", "R_I4_I2", "R_I2_N"], jitter=0.3)
    spawn_stream(w, 6, horizon - 40, 8, "CIV_E1", "civilian", ["R_W_I1", "R_I1_I2", "R_I2_E"], jitter=0.3)
    spawn_stream(w, 8, horizon - 40, 9, "CIV_E2", "civilian", ["R_W_I3", "R_I3_I4", "R_I4_E"], jitter=0.3)

    # Incident 1: blocks eastbound I1->I2 early
    inc1_tick = 60 + int(w.rng.integers(0, 15))
    inc1_end = inc1_tick + 100 + int(w.rng.integers(0, 30))
    schedule_incident(w, at_tick=inc1_tick, incident_id="INC_1",
                      road_id="R_I1_I2", kind="accident", end_tick=inc1_end)

    # Incident 2: blocks northbound I3->I1 later — cascading congestion
    inc2_tick = inc1_tick + 60 + int(w.rng.integers(0, 20))
    inc2_end = inc2_tick + 80 + int(w.rng.integers(0, 20))
    schedule_incident(w, at_tick=inc2_tick, incident_id="INC_2",
                      road_id="R_I3_I1", kind="construction", end_tick=inc2_end)

    # Emergency 1: needs reroute around INC_1
    amb_tick = inc1_tick + 20 + int(w.rng.integers(0, 10))
    spawn(w, at_tick=amb_tick, vid="AMB_1", vtype="ambulance",
          route=["R_W_I1", "R_I1_I2", "R_I2_E"])

    # Emergency 2: needs reroute around INC_2
    fire_tick = inc2_tick + 15 + int(w.rng.integers(0, 10))
    spawn(w, at_tick=fire_tick, vid="FIRE_1", vtype="fire",
          route=["R_S_I3", "R_I3_I1", "R_I1_N"])

    return w


_BUILDERS: dict[str, Callable[[int], World]] = {
    "single_corridor": _build_single_corridor,
    "asymmetric_network": _build_asymmetric_network,
    "incident_and_emergencies": _build_incident_and_emergencies,
    "rush_hour_surge": _build_rush_hour_surge,
    "multi_incident_cascade": _build_multi_incident_cascade,
}
