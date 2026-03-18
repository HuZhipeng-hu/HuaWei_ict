# Repository Guardrails (Demo3 Release)

## Single Source Of Truth

Run commands from the `code/` directory in this repository.
Do not mix scripts from sibling or historical directories.

## Allowed Release Entrypoints

Only these are considered release-path entrypoints:

1. `scripts/collect_event_data.py`
2. `scripts/collect_event_data_continuous.py`
3. `scripts/train_event_model_90_sprint.py`
4. `scripts/finetune_event_onset.py`
5. `scripts/convert_event_onset.py`
6. `scripts/run_event_runtime.py`
7. `scripts/evaluate_event_demo_control.py`

Supporting checks retained by design:

- `scripts/evaluate_ckpt.py`
- `scripts/benchmark_event_runtime_ckpt.py`
- `scripts/validate_event_protocol.py`
- `scripts/preflight.py`

Anything under `experimental/` is outside the release path.

## Control Contract

- public labels: `CONTINUE`, `TENSE_OPEN`, `THUMB_UP`, `WRIST_CW`
- runtime mode: event-driven latch control
- release semantics: `release_mode=command_only`

## Cloud/PI Responsibility Split

- Cloud: prepare, finetune, convert, evaluation
- PI: runtime inference + actuation only
- App transport: out of scope for this repository

## Device Defaults

- Cloud training/conversion defaults: `device_target=Ascend`
- PI runtime default backend: `lite`

## Hygiene Rules

- Do not commit run outputs under `code/artifacts/runs/**`.
- Do not commit split artifacts under `code/artifacts/splits/*.json`.
- Do not commit cache/temp directories (`__pycache__`, `.tmp_pytest*`, `.ipynb_checkpoints`, `pytest-cache-files-*`, `pytest-of-*`).
- Keep docs aligned with the demo3 release path; experimental content must live under `experimental/`.
