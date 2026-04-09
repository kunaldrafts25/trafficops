from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

from models import TrafficOpsAction, TrafficOpsObservation, TrafficOpsState

from .actions import apply_action, expire_plans
from .grading import ScoreBreakdown, grade
from .observations import build_observation
from .sim.engine import tick
from .sim.world import World
from .tasks import build


LLM_PERIOD_TICKS = 10


class TrafficOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._world: Optional[World] = None
        self._state = TrafficOpsState(episode_id=str(uuid4()), step_count=0)
        self._done: bool = False
        self._final_score: Optional[float] = None
        self._final_breakdown: Optional[ScoreBreakdown] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TrafficOpsObservation:
        task = kwargs.get("task", "single_corridor")
        from .tasks import TASK_IDS
        if task not in TASK_IDS:
            task = "single_corridor"
        seed_val = seed if seed is not None else 42

        self._world = build(task, seed=seed_val)
        self._done = False
        self._final_score = None
        self._final_breakdown = None
        self._state = TrafficOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,  # type: ignore[arg-type]
            tick=0,
            horizon=self._world.horizon,
            seed=seed_val,
        )

        return build_observation(
            self._world,
            done=False,
            reward=0.0,
            last_action_error=None,
        )

    def step(
        self,
        action: TrafficOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficOpsObservation:
        assert self._world is not None, "reset() must be called before step()"

        if self._done:
            return build_observation(
                self._world,
                done=True,
                reward=0.0,
                last_action_error="episode_done",
                final_score=self._final_score,
            )

        err = apply_action(self._world, action)

        total_reward = 0.0
        if err is not None:
            total_reward -= 1.0

        for _ in range(LLM_PERIOD_TICKS):
            expire_plans(self._world)
            r, _ = tick(self._world)
            total_reward += r
            if self._episode_terminated():
                break

        self._state.step_count += 1
        self._state.tick = self._world.tick

        self._done = self._episode_terminated()
        final_score = None
        if self._done and self._final_score is None:
            breakdown = grade(self._world)
            self._final_breakdown = breakdown
            self._final_score = breakdown.total
            self._state.final_score = breakdown.total
            self._world.log(
                f"EPISODE_END score={breakdown.total:.3f} breakdown={breakdown.as_dict()}"
            )
            final_score = breakdown.total

        return build_observation(
            self._world,
            done=self._done,
            reward=round(total_reward, 4),
            last_action_error=err,
            final_score=final_score,
        )

    @property
    def state(self) -> TrafficOpsState:
        return self._state

    def grader_breakdown(self) -> Optional[dict]:
        if self._final_breakdown is None:
            return None
        return self._final_breakdown.as_dict()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="TrafficOps",
            description="LLM-supervised adaptive traffic signal control with emergency prioritization and incident response",
            version="0.2.0",
            author="Kunal Singh",
        )

    def close(self) -> None:
        self._world = None
        super().close()

    def _episode_terminated(self) -> bool:
        w = self._world
        if w is None:
            return True
        if w.tick >= w.horizon:
            return True
        if w.metrics.gridlock_events >= 3:
            return True
        nothing_pending = not w.spawn_schedule and not w.incident_schedule
        no_live = all(v.cleared for v in w.vehicles.values())
        if nothing_pending and no_live and w.metrics.spawned_civilian + w.metrics.spawned_emergency > 0:
            return True
        return False
