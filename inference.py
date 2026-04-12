import json
import os
import time

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["grid_balanced", "demand_shift", "incident_corridor", "rush_hour_wave", "multi_crisis"]

MAX_TOTAL_TIME = 17 * 60  # 17 min hard cap (spec says < 20 min)
MAX_TASK_TIME = 3 * 60    # 3 min per task
LLM_TIMEOUT = 25          # 25s per LLM call
MAX_STEPS_PER_TASK = 25

SYSTEM_PROMPT = """You supervise a 4x4 grid of 16 traffic intersections (I_0_0 to I_3_3). Each intersection runs a trained DQN controller that handles routine phase switching. You add value through CITY-LEVEL decisions the local controllers can't make:

ACTIONS (respond with ONE JSON object):
1. set_bias — Make signals favor a direction. Use when one direction has much heavier traffic.
   {"op":"set_bias","targets":["I1","I2","I3"],"params":{"direction":"W","multiplier":2.5,"duration_ticks":100},"reason":"heavy westbound queues"}

2. preempt — Force green for emergency vehicles. Check the EMERGENCIES section for their route, then preempt intersections AHEAD of them.
   {"op":"preempt","targets":["I2","I3"],"params":{"direction":"N","duration_ticks":15},"reason":"ambulance heading north"}

3. reroute — Redirect traffic around a blocked road. Check INCIDENTS for the blocked road, then DETOUR_HINTS for alternative roads.
   {"op":"reroute","targets":["R_I1_I2"],"params":{"blocked_road":"R_I1_I2","detour":["R_I1_I3","R_I3_I4","R_I4_I2"],"duration_ticks":200},"reason":"accident blocks R_I1_I2"}

4. set_coordination — Green wave along a corridor.
   {"op":"set_coordination","targets":["corridor_east"],"params":{"direction":"W","target_speed":0.5,"duration_ticks":100},"reason":"synchronize arterial"}

5. noop — Do nothing when the network is running well.
   {"op":"noop","targets":[],"params":{},"reason":"stable"}

6. cancel — Cancel an active plan by ID.

DECISION GUIDE:
- See INCIDENTS with blocked road? → reroute immediately, use DETOUR_HINTS
- See EMERGENCIES with remaining_route? → preempt intersections along their path
- See high queues in one direction? → set_bias for that direction
- Everything stable? → noop (save your budget)
- Each action except noop/cancel costs 1 from your intervention budget"""


def build_user_message(obs_dict: dict) -> str:
    summary = obs_dict.get("summary", "")
    budget_left = obs_dict.get("interventions_budget", 0) - obs_dict.get("interventions_used", 0)
    parts = [summary]
    if obs_dict.get("last_action_error"):
        parts.append(f"ERROR: {obs_dict['last_action_error']}")
    parts.append(f"Budget: {budget_left}")
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


def run_task(task: str, llm: OpenAI, env, global_start: float):
    env_name = "trafficops"
    print(f"[START] task={task} env={env_name} model={MODEL_NAME}", flush=True)

    task_start = time.time()
    try:
        result = env.reset(task=task)
    except Exception as e:
        # Retry once without task kwarg (fallback to default)
        try:
            result = env.reset(seed=42)
        except Exception as e2:
            print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
            return 0.0
    obs = result.observation
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    step_num = 0
    done = result.done

    while not done:
        # Time guards
        if time.time() - global_start > MAX_TOTAL_TIME:
            print(f"[STEP] step={step_num} action=noop reward=0.00 done=true error=global_timeout", flush=True)
            break
        if time.time() - task_start > MAX_TASK_TIME:
            print(f"[STEP] step={step_num} action=noop reward=0.00 done=true error=task_timeout", flush=True)
            break
        if step_num >= MAX_STEPS_PER_TASK:
            print(f"[STEP] step={step_num} action=noop reward=0.00 done=true error=max_steps", flush=True)
            break

        user_msg = build_user_message(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # Keep message window small — system + last 10 messages
        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                timeout=LLM_TIMEOUT,
            )
            assistant_text = resp.choices[0].message.content or '{"op":"noop","targets":[],"params":{},"reason":"empty"}'
        except Exception:
            assistant_text = '{"op":"noop","targets":[],"params":{},"reason":"api_error"}'

        messages.append({"role": "assistant", "content": assistant_text})
        action_dict = parse_action(assistant_text)

        try:
            from models import TrafficOpsAction
            valid_fields = set(TrafficOpsAction.model_fields.keys())
            clean_dict = {k: v for k, v in action_dict.items() if k in valid_fields}
            action = TrafficOpsAction(**clean_dict)
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
        final_score = obs_dict.get("final_score") or 0.0
        if not final_score:
            import re
            m = re.search(r"EPISODE_END score=([\d.]+)", obs_dict.get("summary", ""))
            if m:
                final_score = float(m.group(1))
    if not final_score:
        final_score = sum(rewards) / max(1, len(rewards)) if rewards else 0.0
        final_score = max(0.0, min(1.0, (final_score + 5.0) / 10.0))

    success = final_score > 0.55
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={step_num} score={final_score:.3f} rewards={rewards_str}",
        flush=True,
    )
    return final_score


def main():
    global_start = time.time()
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    if LOCAL_IMAGE_NAME:
        from client import TrafficOpsEnv
        env = TrafficOpsEnv.from_docker_image(LOCAL_IMAGE_NAME).sync()
    else:
        from client import TrafficOpsEnv
        env = TrafficOpsEnv(base_url=ENV_URL).sync()

    try:
        for task in TASKS:
            if time.time() - global_start > MAX_TOTAL_TIME:
                print(f"[START] task={task} env=trafficops model={MODEL_NAME}", flush=True)
                print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
                continue
            try:
                run_task(task, llm, env, global_start)
            except Exception as e:
                print(f"[START] task={task} env=trafficops model={MODEL_NAME}", flush=True)
                print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
