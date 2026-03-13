# Repository Guardrails (Core 4 Chain)

## Single Source Of Truth

Run all commands from the `code/` directory in this repository.
Do not mix scripts from historical sibling directories.

## Allowed Production Entrypoints

Only these four are considered production chain entrypoints:

1. `scripts/pretrain_db5_repr_method_matrix.py`
2. `scripts/finetune_event_onset.py`
3. `scripts/convert_event_onset.py`
4. `scripts/run_event_runtime.py`

Supporting evaluation scripts retained by design:

- `scripts/evaluate_ckpt.py`
- `scripts/benchmark_event_runtime_ckpt.py`
- `scripts/validate_event_protocol.py`

## Cloud/PI Responsibility Split

- Cloud: pretrain + finetune + convert
- PI: runtime inference + actuation only
- App transport (upload/download): out of scope for this repository

## Device Defaults

- Cloud training/conversion defaults: `device_target=Ascend`
- PI runtime default backend: `lite` (CPU path by runtime config)

## Hygiene Rules

- Do not commit run outputs under `code/artifacts/runs/**`.
- Do not commit cache/temp directories (`__pycache__`, `.tmp_pytest*`, `.ipynb_checkpoints`, `pytest-cache-files-*`, `pytest-of-*`).
- Keep docs aligned with the core 4 chain; remove references to retired legacy scripts.
