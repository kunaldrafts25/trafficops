import json
import os

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["single_corridor", "asymmetric_network", "incident_and_emergencies"]

SYSTEM_PROMPT = """You are an AI traffic operations controller. You manage traffic signals across a road network to maximize throughput, minimize wait times, and prioritize emergency vehicles.

You receive observations describing the network state: intersections (with queue lengths, signal phases, biases), corridors, incidents, emergencies, active plans, and metrics.

You must respond with a single JSON action object. Available operations:
- "noop": Do nothing this step.
- "set_bias": Increase green time for a direction at target intersections. params={"direction": "N"|"S"|"E"|"W", "multiplier": 1.0-10.0, "duration_ticks": int}
- "set_coordination": Coordinate signals along a corridor for green wave. targets=[corridor_id], params={"direction": "N"|"S"|"E"|"W", "target_speed": 0.1-1.0, "duration_ticks": int}
- "preempt": Force green for a direction (for emergencies). targets=[intersection_ids], params={"direction": "N"|"S"|"E"|"W", "duration_ticks": 1-60}
- "reroute": Reroute traffic around a blocked road. targets=[blocked_road], params={"blocked_road": str, "detour": [road_ids], "duration_ticks": int}
- "set_policy": Apply school_zone policy to reduce phase durations. targets=[intersection_ids], params={"policy": "school_zone", "duration_ticks": int}
- "cancel": Cancel an active plan. plan_id=str

Respond ONLY with a JSON object like:
{"op": "set_bias", "targets": ["I1"], "params": {"direction": "W", "multiplier": 2.0, "duration_ticks": 30}, "reason": "heavy westbound queue"}

If unsure, use {"op": "noop", "targets": [], "params": {}, "reason": "observing"}"""


def build_user_message(obs_dict: dict) -> str:
    summary = obs_dict.get("summary", "")
    parts = [summary]
    if obs_dict.get("last_action_error"):
        parts.append(f"\nLAST ERROR: {obs_dict['last_action_error']}")
    budget_left = obs_dict.get("interventions_budget", 0) - obs_dict.get("interventions_used", 0)
    parts.append(f"\nBudget remaining: {budget_left}")
    return "\n".join(parts)


def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"op": "noop", "targets": [], "params": {}, "reason": "parse_failure"}


def run_task(task: str, llm: OpenAI, env):
    env_name = "trafficops"
    print(f"[START] task={task} env={env_name} model={MODEL_NAME}")

    result = env.reset(task=task)
    obs = result.observation
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    step_num = 0
    done = result.done

    while not done:
        user_msg = build_user_message(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        if len(messages) > 20:
            messages = [messages[0]] + messages[-18:]

        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            assistant_text = resp.choices[0].message.content or '{"op":"noop","targets":[],"params":{},"reason":"empty"}'
        except Exception as e:
            assistant_text = '{"op":"noop","targets":[],"params":{},"reason":"api_error"}'
            print(f"[STEP] step={step_num} action=noop reward=0.00 done=false error={str(e)}", flush=True)
            messages.append({"role": "assistant", "content": assistant_text})
            step_num += 1
            continue

        messages.append({"role": "assistant", "content": assistant_text})
        action_dict = parse_action(assistant_text)

        try:
            from models import TrafficOpsAction
            action = TrafficOpsAction(**action_dict)
            result = env.step(action)
            obs = result.observation
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            error_str = obs_dict.get("last_action_error") or "null"
        except Exception as e:
            reward = 0.0
            done = False
            error_str = str(e)

        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action={action_dict.get('op', 'noop')} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}",
            flush=True,
        )
        step_num += 1

    final_score = 0.0
    if isinstance(obs_dict, dict):
        final_score = obs_dict.get("metadata", {}).get("final_score", 0.0) or 0.0
        if not final_score:
            summary = obs_dict.get("summary", "")
            import re
            m = re.search(r"EPISODE_END score=([\d.]+)", summary)
            if m:
                final_score = float(m.group(1))
    if not final_score:
        final_score = sum(rewards) / max(1, len(rewards)) if rewards else 0.0
        final_score = max(0.0, min(1.0, (final_score + 5.0) / 10.0))

    success = final_score > 0.3
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={step_num} score={final_score:.3f} rewards={rewards_str}",
        flush=True,
    )
    return final_score


def main():
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    if LOCAL_IMAGE_NAME:
        from client import TrafficOpsEnv
        env = TrafficOpsEnv.from_docker_image(LOCAL_IMAGE_NAME).sync()
    else:
        from client import TrafficOpsEnv
        env = TrafficOpsEnv(base_url=ENV_URL).sync()

    try:
        for task in TASKS:
            run_task(task, llm, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
