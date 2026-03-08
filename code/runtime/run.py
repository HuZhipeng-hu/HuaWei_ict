"""Runtime entrypoint for prosthesis control."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.control import ProsthesisController
from shared.config import RuntimeConfig, load_runtime_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroGrip Pro V2 runtime")
    parser.add_argument("--config", type=str, default="configs/runtime.yaml", help="Path to runtime YAML config.")
    parser.add_argument("--model", type=str, default=None, help="Override model path from config.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override runtime device ("CPU"/"GPU"/"Ascend", alias "NPU").',
    )
    parser.add_argument("--port", type=str, default=None, help="Override sensor serial port from config.")
    parser.add_argument("--standalone", action="store_true", help="Use standalone sensor/actuator mocks.")
    parser.add_argument("--rate", type=float, default=None, help="Override control loop rate in Hz.")
    parser.add_argument(
        "--infer_rate_hz",
        type=float,
        default=None,
        help="Override inference frequency limit in Hz (0 means no limit).",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=None,
        help="Maximum control-loop cycles to run, then exit gracefully.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NeuroGrip Pro V2 - Runtime Control")
    logger.info("=" * 60)

    config_path = Path(args.config)
    if config_path.exists():
        logger.info("Loading config: %s", config_path)
        config = load_runtime_config(str(config_path))
    else:
        logger.info("Config not found, using default RuntimeConfig.")
        config = RuntimeConfig()

    if args.model:
        config.model_path = args.model
    if args.device:
        config.device.target = args.device
    if args.port:
        config.hardware.sensor_port = args.port
    if args.rate is not None:
        config.control_rate_hz = args.rate
    if args.infer_rate_hz is not None:
        config.infer_rate_hz = args.infer_rate_hz
    if args.standalone:
        config.hardware.sensor_mode = "standalone"
        config.hardware.actuator_mode = "standalone"
        logger.info("Standalone mode enabled (mock sensor + mock actuator).")

    if args.max_cycles is not None and args.max_cycles <= 0:
        raise ValueError("--max_cycles must be > 0 when specified.")

    controller = ProsthesisController(config)
    controller.start(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()

