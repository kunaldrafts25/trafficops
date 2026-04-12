import sys
sys.path.insert(0, ".")

import pytest
from models import TrafficOpsAction
from server.trafficops_environment import TrafficOpsEnvironment
from server.tasks import TASK_IDS


@pytest.fixture
def env():
    return TrafficOpsEnvironment()


class TestDeterminism:
    def test_same_seed_same_score(self, env):
        for task in TASK_IDS:
            scores = []
            for _ in range(2):
                obs = env.reset(seed=42, task=task)
                while not obs.done:
                    obs = env.step(TrafficOpsAction(op="noop"))
                scores.append(obs.final_score)
            assert scores[0] == scores[1], f"{task}: {scores[0]} != {scores[1]}"

    def test_different_seeds_different_scenarios(self, env):
        for task in TASK_IDS:
            scores = []
            for seed in [42, 123, 999]:
                obs = env.reset(seed=seed, task=task)
                while not obs.done:
                    obs = env.step(TrafficOpsAction(op="noop"))
                scores.append(obs.final_score)
            assert not all(s == scores[0] for s in scores), f"{task}: all seeds identical"


class TestGraderRange:
    def test_scores_in_zero_one(self, env):
        for task in TASK_IDS:
            obs = env.reset(seed=42, task=task)
            while not obs.done:
                obs = env.step(TrafficOpsAction(op="noop"))
            bd = env.grader_breakdown()
            for dim, val in bd.items():
                assert 0.0 <= val <= 1.0, f"{task}.{dim} = {val} out of range"

    def test_final_score_on_observation(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert obs.final_score is None
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        assert obs.final_score is not None
        assert 0.0 <= obs.final_score <= 1.0


class TestActions:
    def test_noop_accepted(self, env):
        env.reset(seed=42, task="grid_balanced")
        obs = env.step(TrafficOpsAction(op="noop"))
        assert obs.last_action_error is None

    def test_invalid_target_rejected(self, env):
        env.reset(seed=42, task="grid_balanced")
        obs = env.step(TrafficOpsAction(op="preempt", targets=["FAKE"], params={"direction": "N", "duration_ticks": 10}))
        assert obs.last_action_error is not None

    def test_budget_exhaustion(self, env):
        env.reset(seed=42, task="grid_balanced")
        for i in range(7):
            obs = env.step(TrafficOpsAction(
                op="set_bias", targets=["I_0_0"],
                params={"direction": "W", "multiplier": 2.0, "duration_ticks": 10},
            ))
        assert obs.last_action_error == "budget_exhausted"


class TestRewardSignal:
    def test_smart_beats_noop_on_incident(self, env):
        obs = env.reset(seed=42, task="incident_corridor")
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        noop_score = obs.final_score

        obs = env.reset(seed=42, task="incident_corridor")
        while not obs.done and obs.tick < 55:
            obs = env.step(TrafficOpsAction(op="noop"))
        obs = env.step(TrafficOpsAction(
            op="reroute", targets=["R_h_1_1"],
            params={"blocked_road": "R_h_1_1", "detour": ["R_v_1_1", "R_h_2_1", "R_v_2_2"], "duration_ticks": 200},
        ))
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        smart_score = obs.final_score

        assert smart_score > noop_score, f"smart {smart_score} <= noop {noop_score}"


class TestObservation:
    def test_16_intersections(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert len(obs.intersections) == 16

    def test_40_roads(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert len(obs.roads) == 40

    def test_grid_in_summary(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert "GRID" in obs.summary

    def test_emergency_has_remaining_route(self, env):
        obs = env.reset(seed=42, task="incident_corridor")
        while not obs.done and not obs.emergencies:
            obs = env.step(TrafficOpsAction(op="noop"))
        if obs.emergencies:
            em = obs.emergencies[0]
            assert len(em.remaining_route) > 0

    def test_five_tasks_available(self, env):
        assert len(TASK_IDS) == 5
        for task in TASK_IDS:
            obs = env.reset(seed=42, task=task)
            assert obs.task == task


class TestDQNController:
    def test_dqn_controller_loads(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert env._world.rl_controller is not None

    def test_dqn_controller_mode(self, env):
        obs = env.reset(seed=42, task="grid_balanced")
        assert env._world.controller_mode == "dqn"
