import json
from typing import Any

import gradio as gr


GRID_CSS = """
.traffic-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; max-width: 600px; margin: 0 auto; }
.intersection { border: 2px solid #333; border-radius: 8px; padding: 8px; text-align: center; font-family: monospace; font-size: 12px; min-height: 70px; }
.phase-ns { background: linear-gradient(180deg, #4ade80 0%, #4ade80 50%, #ef4444 50%, #ef4444 100%); }
.phase-ew { background: linear-gradient(90deg, #4ade80 0%, #4ade80 50%, #ef4444 50%, #ef4444 100%); }
.phase-preempt { background: #fbbf24; }
.blocked-road { color: #ef4444; font-weight: bold; }
.emergency-badge { background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
.metric-good { color: #16a34a; }
.metric-bad { color: #dc2626; }
"""


def build_grid_html(obs_dict: dict) -> str:
    intersections = obs_dict.get("intersections", [])
    if not intersections:
        return "<p>No intersection data</p>"

    imap = {i["id"]: i for i in intersections}
    rows = set()
    cols = set()
    for iv in intersections:
        pos = iv.get("position", (0, 0))
        cols.add(pos[0])
        rows.add(pos[1])

    sorted_rows = sorted(rows)
    sorted_cols = sorted(cols)
    n_cols = len(sorted_cols) if sorted_cols else 4

    html = f'<div class="traffic-grid" style="grid-template-columns: repeat({n_cols}, 1fr);">'

    for r in reversed(sorted_rows):
        for c in sorted_cols:
            matching = [iv for iv in intersections if iv.get("position") == [c, r] or iv.get("position") == (c, r)]
            if not matching:
                html += '<div class="intersection" style="background:#f3f4f6;">—</div>'
                continue
            iv = matching[0]
            phase = iv.get("current_phase", "")
            preempt = iv.get("preempt_direction")
            queues = iv.get("queues", {})
            q_n = queues.get("N", 0)
            q_s = queues.get("S", 0)
            q_e = queues.get("E", 0)
            q_w = queues.get("W", 0)

            if preempt:
                css_class = "intersection phase-preempt"
            elif "N" in phase:
                css_class = "intersection phase-ns"
            else:
                css_class = "intersection phase-ew"

            bias_parts = []
            for d, v in iv.get("bias", {}).items():
                if v != 1.0:
                    bias_parts.append(f"{d}x{v:.1f}")
            bias_str = f"<br>bias:{','.join(bias_parts)}" if bias_parts else ""
            preempt_str = f"<br>⚡{preempt}" if preempt else ""

            html += f'''<div class="{css_class}">
                <b>{iv["id"]}</b><br>
                ↑{q_s} ↓{q_n} ←{q_e} →{q_w}
                {preempt_str}{bias_str}
            </div>'''

    html += '</div>'
    return html


def build_status_html(obs_dict: dict) -> str:
    parts = []
    tick = obs_dict.get("tick", 0)
    horizon = obs_dict.get("horizon", 0)
    task = obs_dict.get("task", "")
    budget_used = obs_dict.get("interventions_used", 0)
    budget_total = obs_dict.get("interventions_budget", 0)
    final = obs_dict.get("final_score")

    pct = int(tick / max(1, horizon) * 100)
    parts.append(f"<h3>🚦 {task} — Tick {tick}/{horizon} ({pct}%)</h3>")
    parts.append(f"<div style='background:#e5e7eb;border-radius:4px;height:8px;'><div style='background:#3b82f6;height:8px;border-radius:4px;width:{pct}%;'></div></div>")
    parts.append(f"<p>Budget: {budget_used}/{budget_total} interventions used</p>")

    if final is not None:
        color = "#16a34a" if final > 0.55 else "#dc2626"
        parts.append(f"<h2 style='color:{color}'>Final Score: {final:.4f}</h2>")

    incidents = obs_dict.get("incidents", [])
    active = [i for i in incidents if i.get("active")]
    if active:
        for inc in active:
            parts.append(f"<p class='blocked-road'>🚧 {inc['id']}: {inc['kind']} blocks {inc['road_id']}</p>")

    emergencies = obs_dict.get("emergencies", [])
    live = [e for e in emergencies if not e.get("cleared")]
    if live:
        for em in live:
            parts.append(f"<p><span class='emergency-badge'>🚑 {em['id']}</span> {em['type']} on {em.get('current_road', '?')} → {em['destination']} ETA:{em.get('eta_ticks', '?')}t</p>")
    cleared = [e for e in emergencies if e.get("cleared")]
    if cleared:
        parts.append(f"<p>✅ {len(cleared)} emergency vehicle(s) cleared</p>")

    plans = obs_dict.get("active_plans", [])
    if plans:
        plan_strs = [f"{p['id']}:{p['op']}({','.join(p.get('targets',[]))})" for p in plans]
        parts.append(f"<p>📋 Plans: {', '.join(plan_strs)}</p>")

    m = obs_dict.get("metrics")
    if m:
        parts.append(f"""<table style='font-size:12px;'>
            <tr><td>Cleared</td><td>{m['cleared_civilian']}/{m['spawned_civilian']} civ, {m['cleared_emergency']}/{m['spawned_emergency']} em</td></tr>
            <tr><td>Wait</td><td>avg={m['mean_wait_ticks']:.1f} max={m['max_wait_ticks']}</td></tr>
            <tr><td>Wasted Green</td><td>{m['wasted_green_ticks']}</td></tr>
            <tr><td>Gridlocks</td><td>{m['gridlock_events']}</td></tr>
        </table>""")

    return "\n".join(parts)


def build_trafficops_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    with gr.Blocks(css=GRID_CSS, title="TrafficOps") as demo:
        gr.Markdown("# 🚦 TrafficOps — LLM Traffic Supervisor")
        gr.Markdown("Manage traffic signals across a road network. Bias for demand, preempt for emergencies, reroute around incidents.")

        state = gr.State(value={})

        with gr.Row():
            with gr.Column(scale=2):
                grid_display = gr.HTML(value="<p>Click Reset to start an episode</p>", label="Traffic Grid")
                status_display = gr.HTML(value="", label="Status")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Control Panel")
                    task_dd = gr.Dropdown(
                        choices=["single_corridor", "asymmetric_network", "incident_and_emergencies", "rush_hour_surge", "multi_incident_cascade"],
                        value="single_corridor", label="Task"
                    )
                    seed_input = gr.Number(value=42, label="Seed", precision=0)
                    reset_btn = gr.Button("🔄 Reset", variant="primary")

                with gr.Group():
                    gr.Markdown("### Action")
                    op_dd = gr.Dropdown(
                        choices=["noop", "set_bias", "set_coordination", "preempt", "reroute", "set_policy", "cancel"],
                        value="noop", label="Operation"
                    )
                    targets_input = gr.Textbox(value="", label="Targets (comma-separated)", placeholder="I1,I2,I3")
                    params_input = gr.Textbox(value="{}", label="Params (JSON)", placeholder='{"direction":"W","multiplier":2.5,"duration_ticks":100}')
                    reason_input = gr.Textbox(value="", label="Reason", placeholder="heavy westbound demand")
                    step_btn = gr.Button("▶ Step", variant="secondary")

                obs_json = gr.JSON(label="Raw Observation", visible=False)

        def do_reset(task, seed):
            import requests
            try:
                r = requests.post("http://localhost:8000/reset", json={"seed": int(seed), "task": task}, timeout=10)
                data = r.json()
                obs = data.get("observation", data)
                grid = build_grid_html(obs)
                status = build_status_html(obs)
                return grid, status, obs
            except Exception as e:
                return f"<p>Error: {e}</p>", "", {}

        def do_step(op, targets, params_str, reason, current_obs):
            import requests
            try:
                targets_list = [t.strip() for t in targets.split(",") if t.strip()]
                params = json.loads(params_str) if params_str.strip() else {}
                action = {"op": op, "targets": targets_list, "params": params, "reason": reason}
                r = requests.post("http://localhost:8000/step", json={"action": action}, timeout=10)
                data = r.json()
                obs = data.get("observation", data)
                grid = build_grid_html(obs)
                status = build_status_html(obs)
                return grid, status, obs
            except Exception as e:
                return f"<p>Error: {e}</p>", "", current_obs

        reset_btn.click(do_reset, inputs=[task_dd, seed_input], outputs=[grid_display, status_display, state])
        step_btn.click(do_step, inputs=[op_dd, targets_input, params_input, reason_input, state], outputs=[grid_display, status_display, state])

    return demo
