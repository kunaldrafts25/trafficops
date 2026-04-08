from openenv.core.env_server.http_server import create_app

try:
    from ..models import TrafficOpsAction, TrafficOpsObservation
    from .trafficops_environment import TrafficOpsEnvironment
except (ImportError, ModuleNotFoundError):
    from models import TrafficOpsAction, TrafficOpsObservation
    from server.trafficops_environment import TrafficOpsEnvironment

app = create_app(
    TrafficOpsEnvironment,
    TrafficOpsAction,
    TrafficOpsObservation,
    env_name="trafficops",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
