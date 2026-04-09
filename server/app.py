from openenv.core.env_server.http_server import create_app

try:
    from ..models import TrafficOpsAction, TrafficOpsObservation
    from .trafficops_environment import TrafficOpsEnvironment
    from .gradio_ui import build_trafficops_ui
except (ImportError, ModuleNotFoundError):
    from models import TrafficOpsAction, TrafficOpsObservation
    from server.trafficops_environment import TrafficOpsEnvironment
    from server.gradio_ui import build_trafficops_ui

app = create_app(
    TrafficOpsEnvironment,
    TrafficOpsAction,
    TrafficOpsObservation,
    env_name="trafficops",
    max_concurrent_envs=4,
    gradio_builder=build_trafficops_ui,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
