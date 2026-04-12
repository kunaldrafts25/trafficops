from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TrafficOpsAction, TrafficOpsObservation
except ImportError:
    from models import TrafficOpsAction, TrafficOpsObservation


class TrafficOpsEnv(EnvClient[TrafficOpsAction, TrafficOpsObservation, State]):
    def _step_payload(self, action) -> Dict:
        if hasattr(action, "model_dump"):
            return action.model_dump()
        if isinstance(action, dict):
            return action
        return dict(action)

    def _parse_result(self, payload: Dict) -> StepResult[TrafficOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = TrafficOpsObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
