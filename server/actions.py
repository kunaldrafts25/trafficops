from typing import Optional

from .sim.world import Direction, Plan, World

VALID_DIRS: set[str] = {"N", "S", "E", "W"}


def apply_action(world: World, action) -> Optional[str]:
    op = action.op
    if op == "noop":
        return None
    if op == "cancel":
        return _cancel(world, action.plan_id)

    if world.interventions_used >= world.interventions_budget:
        world.metrics.invalid_actions += 1
        return "budget_exhausted"

    handler = _HANDLERS.get(op)
    if handler is None:
        world.metrics.invalid_actions += 1
        return f"unknown_op:{op}"

    err = handler(world, action)
    if err is not None:
        world.metrics.invalid_actions += 1
        return err

    world.interventions_used += 1
    return None


def expire_plans(world: World) -> None:
    expired = [
        pid
        for pid, p in world.active_plans.items()
        if p.expires_tick is not None and world.tick >= p.expires_tick
    ]
    for pid in expired:
        _revert_plan(world, world.active_plans[pid])
        del world.active_plans[pid]
        world.log(f"PLAN {pid} expired")


def _new_plan(
    world: World,
    op: str,
    targets: list[str],
    params: dict,
    reason: str,
    duration_ticks: Optional[int] = None,
) -> Plan:
    pid = world.new_plan_id()
    expires = world.tick + duration_ticks if duration_ticks is not None else None
    plan = Plan(
        id=pid,
        op=op,
        created_tick=world.tick,
        expires_tick=expires,
        targets=list(targets),
        params=dict(params),
        reason=reason,
    )
    world.active_plans[pid] = plan
    return plan


def _cancel(world: World, plan_id: Optional[str]) -> Optional[str]:
    if not plan_id:
        return "cancel_missing_plan_id"
    plan = world.active_plans.get(plan_id)
    if plan is None:
        return f"cancel_unknown_plan:{plan_id}"
    _revert_plan(world, plan)
    del world.active_plans[plan_id]
    world.log(f"PLAN {plan_id} canceled")
    return None


def _revert_plan(world: World, plan: Plan) -> None:
    op = plan.op
    if op == "preempt":
        for iid in plan.targets:
            I = world.intersections.get(iid)
            if I is not None and I.preempt_direction is not None:
                I.preempt_direction = None
                I.preempt_expires_tick = None
    elif op == "set_bias":
        for iid, prev in plan.snapshot.get("bias", {}).items():
            I = world.intersections.get(iid)
            if I is not None:
                I.bias = dict(prev)
    elif op == "set_coordination":
        cid = plan.targets[0] if plan.targets else None
        if cid and cid in world.corridors:
            world.corridors[cid].coordinated = False
            world.corridors[cid].plan_id = None
            world.corridors[cid].target_speed = None
            world.corridors[cid].phase_offsets = {}
        for iid, prev in plan.snapshot.get("bias", {}).items():
            I = world.intersections.get(iid)
            if I is not None:
                I.bias = dict(prev)
    elif op == "reroute":
        blocked = plan.params.get("blocked_road")
        if blocked in world.reroute_overrides:
            del world.reroute_overrides[blocked]
    elif op == "set_policy":
        for iid, prev in plan.snapshot.get("max_phase_ticks", {}).items():
            I = world.intersections.get(iid)
            if I is not None:
                I.max_phase_ticks = prev


def _op_preempt(world: World, action) -> Optional[str]:
    if not action.targets:
        return "preempt_missing_target"
    direction = action.params.get("direction")
    if direction not in VALID_DIRS:
        return "preempt_bad_direction"
    duration = int(action.params.get("duration_ticks", 15))
    if duration < 1 or duration > 60:
        return "preempt_bad_duration"
    touched: list[str] = []
    for iid in action.targets:
        I = world.intersections.get(iid)
        if I is None:
            return f"preempt_unknown_intersection:{iid}"
        if direction not in I.incoming:
            return f"preempt_no_approach:{iid}/{direction}"
        I.preempt_direction = direction  # type: ignore[assignment]
        I.preempt_expires_tick = world.tick + duration
        touched.append(iid)
    _new_plan(
        world,
        "preempt",
        targets=touched,
        params={"direction": direction, "duration_ticks": duration},
        reason=action.reason,
        duration_ticks=duration,
    )
    world.log(f"PREEMPT {touched} dir={direction} dur={duration}")
    return None


def _op_set_bias(world: World, action) -> Optional[str]:
    if not action.targets:
        return "set_bias_missing_target"
    direction = action.params.get("direction")
    if direction not in VALID_DIRS:
        return "set_bias_bad_direction"
    try:
        multiplier = float(action.params.get("multiplier", 2.0))
    except (TypeError, ValueError):
        return "set_bias_bad_multiplier"
    if multiplier <= 0 or multiplier > 10:
        return "set_bias_multiplier_out_of_range"
    duration = action.params.get("duration_ticks")
    duration_int = int(duration) if duration is not None else None
    snapshot_bias: dict[str, dict] = {}
    for iid in action.targets:
        I = world.intersections.get(iid)
        if I is None:
            return f"set_bias_unknown_intersection:{iid}"
        if direction not in I.incoming:
            return f"set_bias_no_approach:{iid}/{direction}"
        snapshot_bias[iid] = dict(I.bias)
        new_bias = dict(I.bias)
        new_bias[direction] = multiplier  # type: ignore[index]
        I.bias = new_bias
    plan = _new_plan(
        world,
        "set_bias",
        targets=list(action.targets),
        params={"direction": direction, "multiplier": multiplier},
        reason=action.reason,
        duration_ticks=duration_int,
    )
    plan.snapshot["bias"] = snapshot_bias
    world.log(f"SET_BIAS {action.targets} dir={direction} mult={multiplier}")
    return None


def _op_set_coordination(world: World, action) -> Optional[str]:
    if not action.targets:
        return "set_coordination_missing_corridor"
    cid = action.targets[0]
    corridor = world.corridors.get(cid)
    if corridor is None:
        return f"set_coordination_unknown_corridor:{cid}"
    direction = action.params.get("direction", corridor.direction)
    if direction not in VALID_DIRS:
        return "set_coordination_bad_direction"
    try:
        target_speed = float(action.params.get("target_speed", 0.5))
    except (TypeError, ValueError):
        return "set_coordination_bad_speed"
    duration = action.params.get("duration_ticks")
    duration_int = int(duration) if duration is not None else None
    snapshot_bias: dict[str, dict] = {}
    # Compute green-wave offsets: stagger phase starts by road travel time
    cumulative_offset = 0
    offsets: dict[str, int] = {}
    prev_iid = None
    for iid in corridor.intersections:
        I = world.intersections.get(iid)
        if I is None:
            continue
        offsets[iid] = cumulative_offset
        snapshot_bias[iid] = dict(I.bias)
        new_bias = dict(I.bias)
        if direction in I.incoming:
            new_bias[direction] = max(new_bias.get(direction, 1.0), 1.8)  # type: ignore[index]
        I.bias = new_bias
        # Find road from this intersection to next for offset calculation
        if prev_iid is not None:
            for rid, road in world.roads.items():
                if road.from_node == prev_iid and road.to_node == iid:
                    travel_time = int(road.length / max(0.1, target_speed))
                    cumulative_offset += travel_time
                    break
                elif road.from_node == iid and road.to_node == prev_iid:
                    travel_time = int(road.length / max(0.1, target_speed))
                    cumulative_offset += travel_time
                    break
        prev_iid = iid
    corridor.coordinated = True
    corridor.target_speed = target_speed
    corridor.phase_offsets = offsets
    plan = _new_plan(
        world,
        "set_coordination",
        targets=[cid],
        params={"direction": direction, "target_speed": target_speed},
        reason=action.reason,
        duration_ticks=duration_int,
    )
    plan.snapshot["bias"] = snapshot_bias
    corridor.plan_id = plan.id
    world.log(f"COORDINATE corridor={cid} dir={direction} speed={target_speed}")
    return None


def _op_reroute(world: World, action) -> Optional[str]:
    blocked = action.params.get("blocked_road")
    detour = action.params.get("detour")
    if not blocked or blocked not in world.roads:
        return "reroute_bad_blocked_road"
    if not isinstance(detour, list) or not detour:
        return "reroute_missing_detour"
    for rid in detour:
        if rid not in world.roads:
            return f"reroute_unknown_detour_road:{rid}"
    world.reroute_overrides[blocked] = list(detour)

    rerouted_vehicles = 0
    for v in world.vehicles.values():
        if v.cleared:
            continue
        upcoming = v.route[v.route_idx:]
        if blocked in upcoming:
            idx_abs = v.route.index(blocked, v.route_idx)
            v.route = v.route[:idx_abs] + list(detour) + v.route[idx_abs + 1 :]
            rerouted_vehicles += 1

    duration = action.params.get("duration_ticks")
    duration_int = int(duration) if duration is not None else None
    _new_plan(
        world,
        "reroute",
        targets=[blocked],
        params={"blocked_road": blocked, "detour": list(detour)},
        reason=action.reason,
        duration_ticks=duration_int,
    )
    world.log(f"REROUTE around {blocked} via {detour} (in-flight: {rerouted_vehicles})")
    return None


def _op_set_policy(world: World, action) -> Optional[str]:
    policy = action.params.get("policy")
    if policy != "school_zone":
        return "set_policy_unknown_policy"
    if not action.targets:
        return "set_policy_missing_target"
    snapshot: dict[str, int] = {}
    for iid in action.targets:
        I = world.intersections.get(iid)
        if I is None:
            return f"set_policy_unknown_intersection:{iid}"
        snapshot[iid] = I.max_phase_ticks
        I.max_phase_ticks = max(12, I.max_phase_ticks // 2)
    duration = action.params.get("duration_ticks")
    duration_int = int(duration) if duration is not None else None
    plan = _new_plan(
        world,
        "set_policy",
        targets=list(action.targets),
        params={"policy": policy},
        reason=action.reason,
        duration_ticks=duration_int,
    )
    plan.snapshot["max_phase_ticks"] = snapshot
    world.log(f"POLICY {policy} on {action.targets}")
    return None


_HANDLERS = {
    "preempt": _op_preempt,
    "set_bias": _op_set_bias,
    "set_coordination": _op_set_coordination,
    "reroute": _op_reroute,
    "set_policy": _op_set_policy,
}
