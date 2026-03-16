from __future__ import annotations

from types import SimpleNamespace

import scripts.audit_event_model90_pipeline as audit


def _base_args() -> SimpleNamespace:
    return SimpleNamespace(
        screen_loss_types="cross_entropy,cb_focal",
        screen_base_channels="16,24",
        screen_freeze_emg_epochs="5,8",
        screen_encoder_lr_ratios="0.3,0.2",
        screen_pretrained_modes="off,on",
        longrun_seeds="42,52,62",
        runtime_tuning_summary="missing_runtime_tuning_summary.json",
        tune_summary="missing_tune_summary.json",
        target_event_action_accuracy=0.9,
        target_event_action_macro_f1=0.88,
        target_command_success_rate=0.9,
        max_false_trigger_rate=0.05,
        max_false_release_rate=0.05,
    )


def test_param_coverage_blocks_when_neighbor_has_significant_gain() -> None:
    args = _base_args()
    args.screen_loss_types = "cross_entropy"
    screen_summary = {
        "rows": [
            {
                "run_id": "a",
                "loss_type": "cross_entropy",
                "base_channels": 16,
                "freeze_emg_epochs": 5,
                "encoder_lr_ratio": 0.3,
                "pretrained_mode": "off",
            },
            {
                "run_id": "b",
                "loss_type": "cross_entropy",
                "base_channels": 16,
                "freeze_emg_epochs": 5,
                "encoder_lr_ratio": 0.2,
                "pretrained_mode": "off",
            },
            {
                "run_id": "c",
                "loss_type": "cross_entropy",
                "base_channels": 16,
                "freeze_emg_epochs": 8,
                "encoder_lr_ratio": 0.3,
                "pretrained_mode": "off",
            },
            {
                "run_id": "d",
                "loss_type": "cross_entropy",
                "base_channels": 16,
                "freeze_emg_epochs": 8,
                "encoder_lr_ratio": 0.2,
                "pretrained_mode": "off",
            },
            {
                "run_id": "e",
                "loss_type": "cross_entropy",
                "base_channels": 24,
                "freeze_emg_epochs": 5,
                "encoder_lr_ratio": 0.3,
                "pretrained_mode": "off",
            },
            {
                "run_id": "f",
                "loss_type": "cross_entropy",
                "base_channels": 24,
                "freeze_emg_epochs": 5,
                "encoder_lr_ratio": 0.2,
                "pretrained_mode": "off",
            },
            {
                "run_id": "g",
                "loss_type": "cross_entropy",
                "base_channels": 24,
                "freeze_emg_epochs": 8,
                "encoder_lr_ratio": 0.3,
                "pretrained_mode": "off",
            },
            {
                "run_id": "h",
                "loss_type": "cross_entropy",
                "base_channels": 24,
                "freeze_emg_epochs": 8,
                "encoder_lr_ratio": 0.2,
                "pretrained_mode": "off",
            },
        ]
    }
    longrun_summary = {
        "rows": [
            {"run_id": "l1", "candidate_rank": 1, "split_seed": 42},
            {"run_id": "l2", "candidate_rank": 1, "split_seed": 52},
            {"run_id": "l3", "candidate_rank": 1, "split_seed": 62},
        ]
    }
    neighbor_summary = {
        "rows": [{"run_id": "n1"}],
        "event_action_accuracy_gain": 0.02,
        "command_success_rate_gain": 0.01,
        "significant_improvement_found": True,
    }

    result = audit._check_param_coverage(
        args,
        screen_summary=screen_summary,
        longrun_summary=longrun_summary,
        neighbor_summary=neighbor_summary,
    )
    assert result.passed is False
    assert any("significant improvement" in issue for issue in result.issues)


def test_assess_goal_returns_data_bottleneck_when_gates_clear_but_metrics_fail() -> None:
    args = _base_args()
    screen_summary = {
        "rows": [
            {
                "run_id": "s1",
                "event_action_accuracy": 0.72,
                "event_action_macro_f1": 0.70,
                "command_success_rate": 0.48,
                "false_trigger_rate": 0.20,
                "false_release_rate": 0.04,
                "test_accuracy": 0.60,
            }
        ]
    }

    assessment = audit._assess_goal_and_conclusion(
        args,
        data_only_ready=True,
        blocking_issues=[],
        screen_summary=screen_summary,
        longrun_summary={},
        neighbor_summary={},
    )
    assert assessment["conclusion"] == "data_bottleneck_only"
    assert assessment["data_only_bottleneck"] is True
    assert assessment["development_gate"]["passed"] is False
    assert assessment["demo_gate"]["passed"] is False


def test_assess_goal_returns_engineering_not_cleared_when_blocked() -> None:
    args = _base_args()
    assessment = audit._assess_goal_and_conclusion(
        args,
        data_only_ready=False,
        blocking_issues=["implementation: missing run_metadata.json"],
        screen_summary={},
        longrun_summary={},
        neighbor_summary={},
    )
    assert assessment["conclusion"] == "engineering_gates_not_cleared"
    assert assessment["data_only_bottleneck"] is False
