from typing import Optional

from .world import (
    Direction,
    Intersection,
    Road,
    TickStats,
    Vehicle,
    World,
)


GRIDLOCK_STALL_THRESHOLD = 20
HYSTERESIS = 1.15


def phase_serves(intersection: Intersection, direction: Direction) -> bool:
    if intersection.preempt_direction is not None:
        return intersection.preempt_direction == direction
    return direction in intersection.current_phase()


def tick(world: World, on_tick_start=None) -> tuple[float, TickStats]:
    world.tick += 1
    stats = TickStats()

    if on_tick_start is not None:
        on_tick_start(world)

    _activate_incidents(world)
    _spawn_scheduled(world)
    _expire_preempts(world)

    # DQN controller runs every N ticks (decision interval)
    if world.controller_mode == "dqn" and world.rl_controller is not None:
        world.dqn_tick_counter += 1
        if world.dqn_tick_counter >= world.dqn_decision_interval:
            world.dqn_tick_counter = 0
            from .rl_controller import rl_step
            rl_step(world.rl_controller, world)

    for I in world.intersections.values():
        _local_controller_step(I, world, stats)

    _move_vehicles(world, stats)

    if not stats.moved_any and _has_waiting_vehicles(world):
        world.metrics.stalled_streak += 1
        if world.metrics.stalled_streak >= GRIDLOCK_STALL_THRESHOLD:
            stats.gridlock = 1
            world.metrics.gridlock_events += 1
            world.metrics.stalled_streak = 0
            world.log("GRIDLOCK detected, resetting stall counter")
    else:
        world.metrics.stalled_streak = 0

    world.metrics.cleared_civilian += stats.cleared_civ
    world.metrics.cleared_emergency += stats.cleared_em
    world.metrics.wasted_green_ticks += stats.wasted_green

    reward = _compute_tick_reward(world, stats)
    return reward, stats


def _activate_incidents(world: World) -> None:
    still_pending = []
    for ev in world.incident_schedule:
        if ev.tick <= world.tick:
            ev.incident.active = True
            world.roads[ev.incident.road_id].blocked = True
            world.incidents.append(ev.incident)
            world.log(
                f"INCIDENT {ev.incident.id} {ev.incident.kind} blocks {ev.incident.road_id}"
            )
        else:
            still_pending.append(ev)
    world.incident_schedule = still_pending

    for inc in world.incidents:
        if inc.active and inc.end_tick is not None and world.tick >= inc.end_tick:
            inc.active = False
            world.roads[inc.road_id].blocked = False
            world.log(f"INCIDENT {inc.id} cleared on {inc.road_id}")


def _spawn_scheduled(world: World) -> None:
    remaining = []
    for ev in world.spawn_schedule:
        if ev.tick > world.tick:
            remaining.append(ev)
            continue
        route = list(ev.route)
        route = _apply_reroute_overrides(world, route)
        first_road = world.roads.get(route[0]) if route else None
        if first_road is None or first_road.cells[0] is not None or first_road.blocked:
            ev.tick = world.tick + 1
            remaining.append(ev)
            continue
        v = Vehicle(
            id=ev.vehicle_id,
            type=ev.vehicle_type,
            route=route,
            route_idx=0,
            position_in_road=0,
            spawn_tick=world.tick,
        )
        world.vehicles[v.id] = v
        first_road.cells[0] = v.id
        if v.is_emergency():
            world.metrics.spawned_emergency += 1
            world.log(f"SPAWN emergency {v.id} type={v.type} on {first_road.id}")
        else:
            world.metrics.spawned_civilian += 1
    world.spawn_schedule = remaining


def _apply_reroute_overrides(world: World, route: list[str]) -> list[str]:
    for blocked_road, detour in world.reroute_overrides.items():
        if blocked_road in route:
            idx = route.index(blocked_road)
            return route[:idx] + detour + route[idx + 1 :]
    return route


def _expire_preempts(world: World) -> None:
    for I in world.intersections.values():
        if I.preempt_direction is not None and I.preempt_expires_tick is not None:
            if world.tick >= I.preempt_expires_tick:
                I.preempt_direction = None
                I.preempt_expires_tick = None


def _has_nondefault_bias(I: Intersection) -> bool:
    return any(v != 1.0 for v in I.bias.values())


def _get_corridor_offset(iid: str, world: World) -> int:
    for c in world.corridors.values():
        if c.coordinated and iid in c.phase_offsets:
            return c.phase_offsets[iid]
    return 0


def _local_controller_step(I: Intersection, world: World, stats: TickStats) -> None:
    I.phase_timer += 1

    if I.preempt_direction is not None:
        target = I.phase_idx_containing(I.preempt_direction)
        if target is not None and target != I.current_phase_idx:
            I.current_phase_idx = target
            I.phase_timer = 0
        return

    cur_phase = I.current_phase()
    served_demand = 0
    waiting_elsewhere = 0
    pressures: list[float] = []
    for idx, phase in enumerate(I.phases):
        p = 0.0
        for d in phase:
            rid = I.incoming.get(d)
            if rid is None:
                continue
            demand = world.roads[rid].occupancy() + world.roads[rid].queue_at_tail() * 2
            p += I.bias.get(d, 1.0) * demand
            if idx == I.current_phase_idx:
                served_demand += demand
        pressures.append(p)

    for d, rid in I.incoming.items():
        if d not in cur_phase:
            waiting_elsewhere += world.roads[rid].queue_at_tail()

    if served_demand == 0 and waiting_elsewhere > 0:
        stats.wasted_green += 1

    if I.phase_timer < I.min_phase_ticks:
        return

    # Controller hierarchy: LLM bias > Max-Pressure > Fixed-time
    if _has_nondefault_bias(I):
        # LLM has set bias → pressure-responsive with green-wave offsets
        offset = _get_corridor_offset(I.id, world)
        effective_timer = I.phase_timer + offset

        best = max(range(len(I.phases)), key=lambda i: pressures[i])
        current_pressure = pressures[I.current_phase_idx]

        force_switch = effective_timer >= I.max_phase_ticks and waiting_elsewhere > 0
        if force_switch and best == I.current_phase_idx:
            best = (I.current_phase_idx + 1) % len(I.phases)

        if best != I.current_phase_idx and (
            force_switch or pressures[best] > current_pressure * HYSTERESIS
        ):
            I.current_phase_idx = best
            I.phase_timer = 0
    elif world.controller_mode == "max_pressure":
        mp: list[float] = []
        for idx, phase in enumerate(I.phases):
            p = 0.0
            for d in phase:
                upstream = I.incoming.get(d)
                downstream = I.outgoing.get(d)
                u_q = world.roads[upstream].queue_at_tail() if upstream and upstream in world.roads else 0
                d_occ = world.roads[downstream].occupancy() if downstream and downstream in world.roads else 0
                p += max(0, u_q - d_occ)
            mp.append(p)
        best = max(range(len(I.phases)), key=lambda i: mp[i])
        if best != I.current_phase_idx and mp[best] > mp[I.current_phase_idx]:
            I.current_phase_idx = best
            I.phase_timer = 0
        elif I.phase_timer >= I.max_phase_ticks:
            I.current_phase_idx = (I.current_phase_idx + 1) % len(I.phases)
            I.phase_timer = 0
    elif world.controller_mode == "dqn":
        # DQN controller handles switching externally via rl_step()
        # Here we just enforce max_phase as safety fallback
        if I.phase_timer >= I.max_phase_ticks:
            I.current_phase_idx = (I.current_phase_idx + 1) % len(I.phases)
            I.phase_timer = 0
    else:
        # Fixed-time: cycle after max_phase_ticks unconditionally
        if I.phase_timer >= I.max_phase_ticks:
            I.current_phase_idx = (I.current_phase_idx + 1) % len(I.phases)
            I.phase_timer = 0


def _move_vehicles(world: World, stats: TickStats) -> None:
    order = sorted(
        [v for v in world.vehicles.values() if not v.cleared],
        key=lambda v: (-v.route_idx, -v.position_in_road),
    )
    for v in order:
        _try_move(v, world, stats)


def _try_move(v: Vehicle, world: World, stats: TickStats) -> None:
    road = world.roads[v.route[v.route_idx]]

    if v.position_in_road < road.length - 1:
        nxt = v.position_in_road + 1
        if road.cells[nxt] is None:
            road.cells[v.position_in_road] = None
            v.position_in_road = nxt
            road.cells[nxt] = v.id
            stats.moved_any = True
        else:
            v.wait_ticks += 1
            world.metrics.max_wait_ticks_seen = max(
                world.metrics.max_wait_ticks_seen, v.wait_ticks
            )
        return

    if v.route_idx == len(v.route) - 1:
        road.cells[v.position_in_road] = None
        v.cleared = True
        v.clear_tick = world.tick
        stats.moved_any = True
        if v.is_emergency():
            stats.cleared_em += 1
            world.metrics.emergency_clear_times.append(world.tick - v.spawn_tick)
            world.log(f"CLEAR emergency {v.id} in {world.tick - v.spawn_tick} ticks")
        else:
            stats.cleared_civ += 1
        return

    I = world.intersections[road.to_node]
    approach = road.approach_direction
    if not phase_serves(I, approach):
        v.wait_ticks += 1
        world.metrics.max_wait_ticks_seen = max(
            world.metrics.max_wait_ticks_seen, v.wait_ticks
        )
        return

    next_road = world.roads[v.route[v.route_idx + 1]]
    if next_road.blocked or next_road.cells[0] is not None:
        v.wait_ticks += 1
        world.metrics.max_wait_ticks_seen = max(
            world.metrics.max_wait_ticks_seen, v.wait_ticks
        )
        return

    road.cells[v.position_in_road] = None
    v.route_idx += 1
    v.position_in_road = 0
    next_road.cells[0] = v.id
    stats.moved_any = True


def _has_waiting_vehicles(world: World) -> bool:
    return any(not v.cleared for v in world.vehicles.values())


def _compute_tick_reward(world: World, stats: TickStats) -> float:
    r = 0.0
    r += 1.0 * stats.cleared_civ
    r += 5.0 * stats.cleared_em
    total_wait = sum(v.wait_ticks for v in world.vehicles.values() if not v.cleared)
    r -= 0.002 * total_wait
    r -= 0.3 * stats.wasted_green
    r -= 12.0 * stats.gridlock
    return r
