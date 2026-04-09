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
        obs = env.reset(seed=42, task="single_corridor")
        assert obs.final_score is None
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        assert obs.final_score is not None
        assert 0.0 <= obs.final_score <= 1.0


class TestActions:
    def test_noop_accepted(self, env):
        env.reset(seed=42, task="single_corridor")
        obs = env.step(TrafficOpsAction(op="noop"))
        assert obs.last_action_error is None

    def test_invalid_op_rejected(self, env):
        env.reset(seed=42, task="single_corridor")
        obs = env.step(TrafficOpsAction(op="preempt", targets=["NONEXISTENT"], params={"direction": "N", "duration_ticks": 10}))
        assert obs.last_action_error is not None

    def test_budget_exhaustion(self, env):
        env.reset(seed=42, task="single_corridor")
        for i in range(6):
            obs = env.step(TrafficOpsAction(
                op="set_bias",
                targets=["I1"],
                params={"direction": "W", "multiplier": 2.0, "duration_ticks": 10},
            ))
        assert obs.last_action_error == "budget_exhausted"

    def test_preempt_clears_emergency(self, env):
        obs = env.reset(seed=42, task="single_corridor")
        while not obs.done and obs.tick < 70:
            obs = env.step(TrafficOpsAction(op="noop"))
        obs = env.step(TrafficOpsAction(
            op="preempt", targets=["I2"],
            params={"direction": "N", "duration_ticks": 15},
        ))
        assert obs.last_action_error is None


class TestRewardSignal:
    def test_smart_beats_noop(self, env):
        # Noop
        obs = env.reset(seed=42, task="single_corridor")
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        noop_score = obs.final_score

        # Smart: bias arterial + preempt ambulance
        obs = env.reset(seed=42, task="single_corridor")
        obs = env.step(TrafficOpsAction(
            op="set_bias", targets=["I1", "I2", "I3"],
            params={"direction": "W", "multiplier": 2.5, "duration_ticks": 190},
        ))
        while not obs.done and obs.tick < 70:
            obs = env.step(TrafficOpsAction(op="noop"))
        obs = env.step(TrafficOpsAction(
            op="preempt", targets=["I2"],
            params={"direction": "N", "duration_ticks": 15},
        ))
        while not obs.done:
            obs = env.step(TrafficOpsAction(op="noop"))
        smart_score = obs.final_score

        assert smart_score > noop_score, f"smart {smart_score} <= noop {noop_score}"


class TestObservation:
    def test_roads_in_observation(self, env):
        obs = env.reset(seed=42, task="single_corridor")
        assert len(obs.roads) > 0
        r = obs.roads[0]
        assert r.id
        assert r.from_node
        assert r.to_node
        assert r.length > 0

    def test_topology_in_initial_summary(self, env):
        obs = env.reset(seed=42, task="single_corridor")
        assert "TOPOLOGY:" in obs.summary

    def test_emergency_has_remaining_route(self, env):
        obs = env.reset(seed=42, task="incident_and_emergencies")
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
