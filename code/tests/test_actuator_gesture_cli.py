import logging

from event_onset.config import EventRuntimeConfig
from scripts import test_actuator_gesture as cli
from shared.gestures import GestureType


class FakeActuator:
    def __init__(self, *, connect_ok: bool = True):
        self.connect_ok = bool(connect_ok)
        self.connected = False
        self.connect_calls = 0
        self.executed: list[GestureType] = []

    def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = bool(self.connect_ok)
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def execute_gesture(self, gesture: GestureType) -> None:
        self.executed.append(gesture)

    def is_connected(self) -> bool:
        return self.connected

    def get_info(self):
        return {"type": "FakeActuator", "connected": self.connected}


def _runtime_cfg(*, actuator_mode: str = "pca9685") -> EventRuntimeConfig:
    cfg = EventRuntimeConfig()
    cfg.hardware.actuator_mode = actuator_mode
    return cfg


def test_command_mapping_and_safe_relax(monkeypatch):
    actuator = FakeActuator()
    commands = iter(["f", "p", "q"])

    monkeypatch.setattr(cli, "load_event_runtime_config", lambda _path: _runtime_cfg())
    monkeypatch.setattr(cli, "create_actuator", lambda _hw: actuator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    rc = cli.main(["--strict_hardware", "false"])

    assert rc == 0
    assert actuator.executed == [
        GestureType.RELAX,
        GestureType.FIST,
        GestureType.PINCH,
        GestureType.RELAX,
    ]


def test_invalid_command_does_not_trigger_gesture(monkeypatch, caplog):
    actuator = FakeActuator()
    commands = iter(["x", "q"])

    monkeypatch.setattr(cli, "load_event_runtime_config", lambda _path: _runtime_cfg())
    monkeypatch.setattr(cli, "create_actuator", lambda _hw: actuator)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    caplog.set_level(logging.WARNING)

    rc = cli.main(["--strict_hardware", "false"])

    assert rc == 0
    assert actuator.executed == [GestureType.RELAX, GestureType.RELAX]
    assert "Unknown command" in caplog.text


def test_keyboard_interrupt_exits_and_runs_safe_shutdown(monkeypatch):
    actuator = FakeActuator()

    def _raise_interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_event_runtime_config", lambda _path: _runtime_cfg())
    monkeypatch.setattr(cli, "create_actuator", lambda _hw: actuator)
    monkeypatch.setattr("builtins.input", _raise_interrupt)

    rc = cli.main(["--strict_hardware", "false"])

    assert rc == 0
    assert actuator.executed == [GestureType.RELAX, GestureType.RELAX]


def test_strict_mode_fails_when_smbus_missing(monkeypatch):
    actuator = FakeActuator()

    monkeypatch.setattr(cli, "load_event_runtime_config", lambda _path: _runtime_cfg())
    monkeypatch.setattr(cli, "create_actuator", lambda _hw: actuator)
    monkeypatch.setattr(cli, "PCA9685Actuator", FakeActuator)
    monkeypatch.setattr(cli.pca9685_actuator, "SMBUS_AVAILABLE", False)

    rc = cli.main([])

    assert rc == 1
    assert actuator.connect_calls == 0


def test_strict_mode_fails_on_connect_error(monkeypatch):
    actuator = FakeActuator(connect_ok=False)

    monkeypatch.setattr(cli, "load_event_runtime_config", lambda _path: _runtime_cfg())
    monkeypatch.setattr(cli, "create_actuator", lambda _hw: actuator)
    monkeypatch.setattr(cli, "PCA9685Actuator", FakeActuator)
    monkeypatch.setattr(cli.pca9685_actuator, "SMBUS_AVAILABLE", True)

    rc = cli.main([])

    assert rc == 1
    assert actuator.connect_calls == 1
