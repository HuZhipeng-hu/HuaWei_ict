# NeuroGrip Event-Onset v1 (Final Mainline)

This branch is **event-onset only** and the recommended workflow is now a single one-click command:

`DB5 aligned pretrain -> wearer finetune A/B (scratch vs pretrained) -> auto select best -> MindIR convert -> CKPT/Lite benchmark gate`

## 1) One-Click Pipeline (Recommended)

```bash
python scripts/train_event_pipeline.py \
  --db5_data_dir ../data_ninaproDB5 \
  --wearer_data_dir ../data \
  --budget_per_class 60 \
  --device_target Ascend \
  --device_id 0 \
  --run_id event_pipeline_v1
```

Main outputs:

- `artifacts/runs/<run_id>/final_selection.json`
- `artifacts/runs/<run_id>/final_artifacts.json`
- `artifacts/runs/<run_id>/models/event_onset_selected.mindir`
- `artifacts/runs/<run_id>/models/event_onset_selected.model_metadata.json`

Notes:

- Budget mode defaults to `60` train windows per class (`RELAX/FIST/PINCH`).
- A/B selection rule: pretrained must **strictly outperform** scratch (`macro_f1`, then `accuracy` tie-break), otherwise scratch is chosen automatically.
- Production deployment format is `MindIR`; CKPT is debug only.

## 2) Preflight

```bash
python scripts/preflight.py --mode local
python scripts/preflight.py --mode ascend
```

Optional explicit paths:

```bash
python scripts/preflight.py \
  --mode ascend \
  --db5_data_dir ../data_ninaproDB5 \
  --wearer_data_dir ../data \
  --budget_per_class 60
```

## 3) Runtime on Orange Pi

Production backend (`lite`):

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite
```

Debug backend (`ckpt`, offline troubleshooting only):

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend ckpt
```

## 4) Hardware Gesture Test (No Inference)

Use this mode to validate actuator behavior directly from keyboard input:

```bash
python scripts/test_actuator_gesture.py \
  --config configs/runtime_event_onset.yaml
```

Commands:

- `r` -> `RELAX`
- `f` -> `FIST`
- `p` -> `PINCH`
- `o` -> `OK`
- `y` -> `YE`
- `s` -> `SIDEGRIP`
- `i` -> print actuator info
- `h` -> help
- `q` -> quit

Safety behavior:

- startup auto `RELAX`
- exit auto `RELAX` then disconnect

## 5) Advanced Debug (Step-by-Step, Optional)

Not recommended for daily usage, but kept for troubleshooting:

- `scripts/pretrain_ninapro_db5.py` (`--pretrain_mode legacy53` only for debug fallback)
- `scripts/finetune_event_onset.py`
- `scripts/convert_event_onset.py`
- `scripts/benchmark_event_runtime_ckpt.py`
- `scripts/evaluate_ckpt.py`
