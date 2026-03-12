"""Interactive hardware-only actuator gesture tester."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_runtime_config
from runtime.hardware import pca9685_actuator
from runtime.hardware.factory import create_actuator
from runtime.hardware.pca9685_actuator import PCA9685Actuator
from shared.gestures import GestureType


LOGGER = logging.getLogger("actuator_gesture_test")

COMMAND_TO_GESTURE = {
    "r": GestureType.RELAX,
    "f": GestureType.FIST,
    "p": GestureType.PINCH,
    "o": GestureType.OK,
    "y": GestureType.YE,
    "s": GestureType.SIDEGRIP,
}

HELP_TEXT = """\
Commands:
  r -> RELAX
  f -> FIST
  p -> PINCH
  o -> OK
  y -> YE
  s -> SIDEGRIP
  i -> actuator info
  h -> help
  q -> quit
"""


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw!r}")


def _resolve_log_level(raw: str) -> int:
    level = getattr(logging, str(raw).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {raw!r}")
    return level


def _parse_command(raw: str) -> tuple[str, GestureType | str | None]:
    token = raw.strip().lower()
    if not token:
        return "empty", None
    token = token.split()[0]
    if token in COMMAND_TO_GESTURE:
        return "gesture", COMMAND_TO_GESTURE[token]
    if token in {"q", "quit", "exit"}:
        return "quit", None
    if token in {"h", "help", "?"}:
        return "help", None
    if token in {"i", "info"}:
        return "info", None
    return "unknown", token


def _is_connected(actuator) -> bool:
    if actuator is None or not hasattr(actuator, "is_connected"):
        return False
    try:
        return bool(actuator.is_connected())
    except Exception:
        return False


def _validate_strict_hardware(runtime_cfg, actuator) -> None:
    mode = str(runtime_cfg.hardware.actuator_mode).strip().lower()
    if mode != "pca9685":
        raise RuntimeError(
            "strict_hardware=true requires hardware.actuator_mode=pca9685 "
            f"(got {runtime_cfg.hardware.actuator_mode!r})."
        )
    if not isinstance(actuator, PCA9685Actuator):
        raise RuntimeError("strict_hardware=true requires PCA9685 actuator implementation.")
    if not pca9685_actuator.SMBUS_AVAILABLE:
        raise RuntimeError(
            "strict_hardware=true rejected mock fallback: smbus2 is not available. "
            "Install smbus2 on the target device."
        )


def _safe_relax_and_disconnect(actuator) -> None:
    if actuator is None:
        return

    if _is_connected(actuator):
        try:
            actuator.execute_gesture(GestureType.RELAX)
            LOGGER.info("Safety pose applied on shutdown: RELAX")
        except Exception as exc:
            LOGGER.warning("Failed to apply RELAX during shutdown: %s", exc)

    try:
        actuator.disconnect()
        LOGGER.info("Actuator disconnected.")
    except Exception as exc:
        LOGGER.warning("Failed to disconnect actuator cleanly: %s", exc)


def run_interactive_loop(actuator, input_fn: Callable[[str], str] | None = None) -> int:
    if input_fn is None:
        input_fn = input
    while True:
        try:
            raw = input_fn("gesture-test> ")
        except EOFError:
            LOGGER.info("EOF received. Exiting.")
            return 0
        except KeyboardInterrupt:
            print("")
            LOGGER.info("Interrupted by user. Exiting.")
            return 0

        action, payload = _parse_command(raw)
        if action == "empty":
            continue
        if action == "help":
            LOGGER.info("\n%s", HELP_TEXT)
            continue
        if action == "info":
            LOGGER.info("Actuator info: %s", actuator.get_info())
            continue
        if action == "quit":
            LOGGER.info("Exit requested.")
            return 0
        if action == "gesture":
            gesture = payload
            assert isinstance(gesture, GestureType)
            actuator.execute_gesture(gesture)
            LOGGER.info("Executed gesture: %s", gesture.name)
            continue

        LOGGER.warning("Unknown command: %s (type 'h' for help)", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive hardware gesture tester (no model inference).")
    parser.add_argument("--config", default="configs/runtime_event_onset.yaml")
    parser.add_argument(
        "--strict_hardware",
        type=_parse_bool,
        default=True,
        help="Whether to reject mock hardware fallback (true/false). Default: true.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        level = _resolve_log_level(args.log_level)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    actuator = None
    try:
        runtime_cfg = load_event_runtime_config(args.config)
        actuator = create_actuator(runtime_cfg.hardware)

        LOGGER.info("Hardware gesture test mode started (no model inference).")
        LOGGER.info("Configured actuator mode: %s", runtime_cfg.hardware.actuator_mode)
        if args.strict_hardware:
            _validate_strict_hardware(runtime_cfg, actuator)
            LOGGER.info("Strict hardware check passed: PCA9685 + smbus2 available.")
        else:
            LOGGER.warning("strict_hardware=false; mock actuator fallback is allowed for debugging.")

        if not actuator.connect():
            raise RuntimeError("Actuator connection failed.")

        LOGGER.info("Actuator connected.")
        LOGGER.info("Actuator info: %s", actuator.get_info())

        actuator.execute_gesture(GestureType.RELAX)
        LOGGER.info("Safety pose applied at startup: RELAX")
        LOGGER.info("\n%s", HELP_TEXT)
        return run_interactive_loop(actuator)
    except Exception as exc:
        LOGGER.error("Hardware gesture test failed: %s", exc)
        return 1
    finally:
        _safe_relax_and_disconnect(actuator)


if __name__ == "__main__":
    raise SystemExit(main())
