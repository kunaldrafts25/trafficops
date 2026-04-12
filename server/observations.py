from typing import Optional

from models import (
    CorridorView,
    EmergencyView,
    IncidentView,
    IntersectionView,
    MetricsView,
    PlanView,
    RoadView,
    TrafficOpsObservation,
)

from .sim.world import EMERGENCY_TYPES, World


def build_observation(
    world: World,
    done: bool,
    reward: float,
    last_action_error: Optional[str],
    final_score: Optional[float] = None,
) -> TrafficOpsObservation:
    roads = [_view_road(world, rid) for rid in sorted(world.roads)]
    intersections = [_view_intersection(world, iid) for iid in sorted(world.intersections)]
    corridors = [_view_corridor(world, cid) for cid in sorted(world.corridors)]
    incidents = [_view_incident(inc) for inc in world.incidents]
    emergencies = _view_emergencies(world)
    plans = [_view_plan(p) for p in world.active_plans.values()]
    metrics = _view_metrics(world)

    summary = _build_summary(
        world, intersections, incidents, emergencies, plans, metrics, last_action_error
    )

    return TrafficOpsObservation(
        task=world.task,  # type: ignore[arg-type]
        tick=world.tick,
        horizon=world.horizon,
        summary=summary,
        roads=roads,
        intersections=intersections,
        corridors=corridors,
        incidents=incidents,
        emergencies=emergencies,
        active_plans=plans,
        metrics=metrics,
        interventions_used=world.interventions_used,
        interventions_budget=world.interventions_budget,
        last_action_error=last_action_error,
        final_score=final_score,
        done=done,
        reward=reward,
    )


def _view_road(world: World, rid: str) -> RoadView:
    r = world.roads[rid]
    return RoadView(
        id=r.id,
        from_node=r.from_node,
        to_node=r.to_node,
        approach_direction=r.approach_direction,
        length=r.length,
        occupancy=r.occupancy(),
        queue_at_stop=r.queue_at_tail(),
        blocked=r.blocked,
    )


def _view_intersection(world: World, iid: str) -> IntersectionView:
    I = world.intersections[iid]
    queues: dict = {}
    for d in ("N", "S", "E", "W"):
        rid = I.incoming.get(d)  # type: ignore[arg-type]
        queues[d] = world.roads[rid].queue_at_tail() if rid else 0
    bias = {d: I.bias.get(d, 1.0) for d in ("N", "S", "E", "W")}  # type: ignore[misc]
    return IntersectionView(
        id=I.id,
        position=I.position,
        current_phase=I.phase_name(),
        phase_timer=I.phase_timer,
        min_phase_ticks=I.min_phase_ticks,
        max_phase_ticks=I.max_phase_ticks,
        queues=queues,
        bias=bias,
        preempt_direction=I.preempt_direction,
        preempt_expires_tick=I.preempt_expires_tick,
        neighbors=list(I.neighbors),
    )


def _view_corridor(world: World, cid: str) -> CorridorView:
    c = world.corridors[cid]
    total_flow = 0
    for iid in c.intersections:
        I = world.intersections.get(iid)
        if I is None:
            continue
        for d, rid in I.incoming.items():
            if d == c.direction:
                total_flow += world.roads[rid].occupancy()
    return CorridorView(
        id=c.id,
        intersections=list(c.intersections),
        direction=c.direction,
        coordinated=c.coordinated,
        plan_id=c.plan_id,
        target_speed=c.target_speed,
        total_flow=total_flow,
    )


def _view_incident(inc) -> IncidentView:
    return IncidentView(
        id=inc.id,
        road_id=inc.road_id,
        kind=inc.kind,
        start_tick=inc.start_tick,
        end_tick=inc.end_tick,
        active=inc.active,
    )


def _view_emergencies(world: World) -> list[EmergencyView]:
    out: list[EmergencyView] = []
    for v in world.vehicles.values():
        if v.type not in EMERGENCY_TYPES:
            continue
        current_road = v.route[v.route_idx] if v.route_idx < len(v.route) else None
        remaining = v.route[v.route_idx:] if not v.cleared else []
        eta = _estimate_eta(world, v) if not v.cleared else 0
        out.append(
            EmergencyView(
                id=v.id,
                type=v.type,  # type: ignore[arg-type]
                origin=world.roads[v.route[0]].from_node,
                destination=world.roads[v.route[-1]].to_node,
                current_road=current_road,
                remaining_route=remaining,
                ticks_since_spawn=world.tick - v.spawn_tick,
                eta_ticks=eta,
                cleared=v.cleared,
            )
        )
    return out


def _estimate_eta(world: World, v) -> int:
    if v.cleared:
        return 0
    remaining = 0
    for idx in range(v.route_idx, len(v.route)):
        road = world.roads[v.route[idx]]
        if idx == v.route_idx:
            remaining += road.length - v.position_in_road
        else:
            remaining += road.length
    return remaining


def _view_plan(p) -> PlanView:
    return PlanView(
        id=p.id,
        op=p.op,
        created_tick=p.created_tick,
        expires_tick=p.expires_tick,
        targets=list(p.targets),
        params=dict(p.params),
    )


def _view_metrics(world: World) -> MetricsView:
    m = world.metrics
    cleared_times = m.emergency_clear_times
    mean_wait = 0.0
    live = [v for v in world.vehicles.values() if not v.cleared]
    if live:
        mean_wait = sum(v.wait_ticks for v in live) / len(live)
    total_queue = sum(r.occupancy() for r in world.roads.values())
    return MetricsView(
        cleared_civilian=m.cleared_civilian,
        cleared_emergency=m.cleared_emergency,
        spawned_civilian=m.spawned_civilian,
        spawned_emergency=m.spawned_emergency,
        mean_wait_ticks=round(mean_wait, 2),
        max_wait_ticks=m.max_wait_ticks_seen,
        total_queue=total_queue,
        wasted_green_ticks=m.wasted_green_ticks,
        gridlock_events=m.gridlock_events,
        conflicting_plans=m.invalid_actions,
    )


def _build_summary(
    world: World,
    intersections: list[IntersectionView],
    incidents: list[IncidentView],
    emergencies: list[EmergencyView],
    plans: list[PlanView],
    metrics: MetricsView,
    last_action_error: Optional[str],
) -> str:
    lines: list[str] = []
    budget_left = world.interventions_budget - world.interventions_used
    lines.append(
        f"Tick {world.tick}/{world.horizon} task={world.task} interventions={world.interventions_used}/{world.interventions_budget} (budget_left={budget_left})"
    )

    active_incs = [i for i in incidents if i.active]
    if active_incs:
        parts = [f"{i.id}({i.kind}) blocks {i.road_id}" for i in active_incs]
        lines.append("INCIDENTS: " + "; ".join(parts))
        blocked_roads = [r.id for r in world.roads.values() if r.blocked]
        if blocked_roads:
            alt_routes = []
            for rid in blocked_roads:
                road = world.roads[rid]
                alts = [r.id for r in world.roads.values()
                        if r.from_node == road.from_node and r.id != rid and not r.blocked]
                if alts:
                    alt_routes.append(f"{rid} alt_from_{road.from_node}=[{','.join(alts)}]")
            if alt_routes:
                lines.append("DETOUR_HINTS: " + "; ".join(alt_routes))

    live_em = [e for e in emergencies if not e.cleared]
    if live_em:
        parts = [
            f"{e.id}({e.type}) on {e.current_road} route={e.remaining_route} dest={e.destination} eta~{e.eta_ticks}t"
            for e in live_em
        ]
        lines.append("EMERGENCIES: " + "; ".join(parts))
    elif metrics.spawned_emergency > 0:
        lines.append(f"EMERGENCIES: all {metrics.cleared_emergency} cleared")

    if plans:
        parts = [
            f"{p.id}:{p.op}({','.join(p.targets) or '-'})exp={p.expires_tick}"
            for p in plans
        ]
        lines.append("ACTIVE_PLANS: " + "; ".join(parts))
    else:
        lines.append("ACTIVE_PLANS: none")

    # Compact grid format for large networks
    imap = {iv.id: iv for iv in intersections}
    rows = sorted(set(iv.position[1] for iv in intersections), reverse=True)
    cols = sorted(set(iv.position[0] for iv in intersections))
    if len(intersections) > 6:
        grid_lines = ["GRID (phase S_queue/W_queue):"]
        for r in rows:
            cells = []
            for c in cols:
                matching = [iv for iv in intersections if iv.position == (c, r) or iv.position == [c, r]]
                if not matching:
                    cells.append("  ---  ")
                    continue
                iv = matching[0]
                ph = "NS" if "N" in iv.current_phase else "EW"
                if iv.preempt_direction:
                    ph = f"!{iv.preempt_direction}"
                sq = iv.queues.get("S", 0)
                wq = iv.queues.get("W", 0)
                cells.append(f"[{ph} {sq}/{wq}]")
            grid_lines.append("  " + " ".join(cells))
        lines.append("\n".join(grid_lines))
    else:
        int_desc = []
        for iv in intersections:
            queues = iv.queues
            qstr = " ".join(f"{d}={queues.get(d,0)}" for d in ("N", "S", "E", "W") if queues.get(d, 0) > 0)
            tag = iv.current_phase
            if iv.preempt_direction:
                tag += f"!preempt={iv.preempt_direction}"
            bias_tags = [f"{d}x{v:.1f}" for d, v in iv.bias.items() if v != 1.0]
            bias_str = f" bias=[{','.join(bias_tags)}]" if bias_tags else ""
            int_desc.append(f"{iv.id}[{tag} q({qstr or '-'}){bias_str}]")
        lines.append("NETWORK: " + " ".join(int_desc))

    lines.append(
        f"METRICS: cleared_civ={metrics.cleared_civilian}/{metrics.spawned_civilian} "
        f"cleared_em={metrics.cleared_emergency}/{metrics.spawned_emergency} "
        f"mean_wait={metrics.mean_wait_ticks} max_wait={metrics.max_wait_ticks} "
        f"wasted_green={metrics.wasted_green_ticks} gridlocks={metrics.gridlock_events}"
    )

    if world.tick == 0:
        if len(world.roads) <= 15:
            topo = []
            for rid in sorted(world.roads):
                r = world.roads[rid]
                topo.append(f"{rid}:{r.from_node}->{r.to_node}(len={r.length})")
            lines.append("TOPOLOGY: " + " ".join(topo))
        else:
            lines.append(f"GRID: {len(world.intersections)} intersections, {len(world.roads)} roads")
            lines.append("Road naming: R_h_{{row}}_{{col}} (horizontal W->E), R_v_{{row}}_{{col}} (vertical S->N)")
            lines.append("Corridors: ew_row_0..3 (east-west), ns_col_0..3 (north-south)")

    recent = world.event_log[-5:]
    if recent:
        lines.append("RECENT: " + " | ".join(recent))

    if last_action_error:
        lines.append(f"LAST_ACTION_ERROR: {last_action_error}")

    return "\n".join(lines)
