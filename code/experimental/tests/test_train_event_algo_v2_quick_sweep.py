from __future__ import annotations

from experimental.scripts.train_event_algo_v2_quick_sweep import _build_presets, _pick_metric, _rank_key


def test_pick_metric_handles_none_and_invalid() -> None:
    assert _pick_metric(None, "bad", 0.25, default=0.0) == 0.25
    assert _pick_metric(None, "bad", default=0.3) == 0.3


def test_rank_key_uses_alias_fields() -> None:
    row_test_prefix = {
        "test_command_success_rate": 0.60,
        "test_false_trigger_rate": 0.10,
        "test_false_release_rate": 0.08,
        "event_action_accuracy": 0.70,
        "event_action_macro_f1": 0.66,
    }
    row_plain_prefix = {
        "command_success_rate": 0.61,
        "false_trigger_rate": 0.10,
        "false_release_rate": 0.08,
        "event_action_accuracy": 0.70,
        "event_action_macro_f1": 0.66,
    }
    ranked = sorted([row_test_prefix, row_plain_prefix], key=_rank_key, reverse=True)
    assert ranked[0] is row_plain_prefix


def test_presets_include_targeted_pairs() -> None:
    presets = _build_presets()
    names = {item["preset"] for item in presets}
    assert "ccw_protect" in names
    assert "release_guard" in names
    assert len(presets) == 6
