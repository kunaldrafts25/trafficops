from typing import Callable

from .sim.builders import (
    add_corridor,
    add_intersection,
    add_road,
    connect_neighbors,
    new_world,
    spawn,
    spawn_stream,
    wire,
)
from .sim.world import World


TASK_IDS = ["single_corridor", "asymmetric_network", "incident_and_emergencies"]


def build(task: str, seed: int) -> World:
    builder = _BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"unknown task: {task}")
    return builder(seed)


def _build_single_corridor(seed: int) -> World:
    horizon = 160
    w = new_world("single_corridor", horizon=horizon, seed=seed, interventions_budget=5)

    for i, x in enumerate([0, 1, 2], start=1):
        add_intersection(w, f"I{i}", position=(x, 0), min_phase_ticks=8, max_phase_ticks=40)

    arterial_len = 7
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
                 vid_prefix="ART", vtype="civilian", route=arterial_route)

    for iid in ("I1", "I2", "I3"):
        route = [f"R_N_{iid}", f"R_{iid}_S"]
        spawn_stream(w, start_tick=5, end_tick=horizon - 20, period=18,
                     vid_prefix=f"CROSS_{iid}", vtype="civilian", route=route)

    return w


def _build_asymmetric_network(seed: int) -> World:
    horizon = 240
    w = new_world("asymmetric_network", horizon=horizon, seed=seed, interventions_budget=6)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=6, max_phase_ticks=40)

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

    phase_a_end = 130

    spawn_stream(w, 2, phase_a_end, 4, "NA_W", "civilian", ["R_SW_I3", "R_I3_I1", "R_I1_N"])
    spawn_stream(w, 3, phase_a_end, 4, "NA_E", "civilian", ["R_SE_I4", "R_I4_I2", "R_I2_N"])
    spawn_stream(w, 6, phase_a_end, 22, "XA_N", "civilian", ["R_WW_I1", "R_I1_I2", "R_I2_E"])
    spawn_stream(w, 8, phase_a_end, 22, "XA_S", "civilian", ["R_WW_I3", "R_I3_I4", "R_I4_E"])

    spawn_stream(w, phase_a_end, horizon - 20, 22, "NB_W", "civilian", ["R_SW_I3", "R_I3_I1", "R_I1_N"])
    spawn_stream(w, phase_a_end + 1, horizon - 20, 22, "NB_E", "civilian", ["R_SE_I4", "R_I4_I2", "R_I2_N"])
    spawn_stream(w, phase_a_end + 2, horizon - 20, 4, "EB_N", "civilian", ["R_WW_I1", "R_I1_I2", "R_I2_E"])
    spawn_stream(w, phase_a_end + 3, horizon - 20, 4, "EB_S", "civilian", ["R_WW_I3", "R_I3_I4", "R_I4_E"])

    return w


def _build_incident_and_emergencies(seed: int) -> World:
    horizon = 280
    w = new_world("incident_and_emergencies", horizon=horizon, seed=seed, interventions_budget=8)

    positions = {"I1": (0, 1), "I2": (1, 1), "I3": (0, 0), "I4": (1, 0)}
    for iid, pos in positions.items():
        add_intersection(w, iid, position=pos, min_phase_ticks=6, max_phase_ticks=40)

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

    spawn_stream(w, 2, horizon - 40, 8, "CIV_N1", "civilian", ["R_S_I3", "R_I3_I1", "R_I1_N"])
    spawn_stream(w, 4, horizon - 40, 9, "CIV_N2", "civilian", ["R_S_I4", "R_I4_I2", "R_I2_N"])
    spawn_stream(w, 6, horizon - 40, 9, "CIV_E1", "civilian", ["R_W_I1", "R_I1_I2", "R_I2_E"])
    spawn_stream(w, 8, horizon - 40, 10, "CIV_E2", "civilian", ["R_W_I3", "R_I3_I4", "R_I4_E"])

    from .sim.builders import schedule_incident
    schedule_incident(w, at_tick=60, incident_id="INC_1",
                      road_id="R_I1_I2", kind="accident", end_tick=240)

    spawn(w, at_tick=90, vid="AMB_1", vtype="ambulance",
          route=["R_S_I3", "R_I3_I1", "R_I1_I2", "R_I2_E"])

    spawn(w, at_tick=135, vid="FIRE_1", vtype="fire",
          route=["R_S_I4", "R_I4_I2", "R_I2_N"])

    spawn(w, at_tick=170, vid="POLICE_1", vtype="police",
          route=["R_W_I1", "R_I1_I2", "R_I2_E"])

    return w


_BUILDERS: dict[str, Callable[[int], World]] = {
    "single_corridor": _build_single_corridor,
    "asymmetric_network": _build_asymmetric_network,
    "incident_and_emergencies": _build_incident_and_emergencies,
}
