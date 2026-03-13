from __future__ import annotations

from scripts.finetune_event_onset import _resolve_recordings_manifest_path


def test_resolve_recordings_manifest_path_uses_explicit_arg(tmp_path) -> None:
    manifest = tmp_path / "explicit.csv"
    manifest.write_text("relative_path\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    resolved = _resolve_recordings_manifest_path(
        data_dir=str(tmp_path),
        config_path=str(config),
        manifest_arg=str(manifest),
    )
    assert resolved.endswith("explicit.csv")


def test_resolve_recordings_manifest_path_uses_config_default(tmp_path) -> None:
    manifest = tmp_path / "recordings_manifest.csv"
    manifest.write_text("relative_path\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    resolved = _resolve_recordings_manifest_path(
        data_dir=str(tmp_path),
        config_path=str(config),
        manifest_arg=None,
    )
    assert resolved.endswith("recordings_manifest.csv")


def test_resolve_recordings_manifest_path_missing_raises_clear_error(tmp_path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    try:
        _resolve_recordings_manifest_path(
            data_dir=str(tmp_path),
            config_path=str(config),
            manifest_arg=None,
        )
    except FileNotFoundError as exc:
        assert "Event-onset finetune requires recordings manifest" in str(exc)
        return
    raise AssertionError("expected FileNotFoundError")
