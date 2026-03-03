# NeuroGrip Handoff Runbook

This runbook is for handing off the project to a machine that has the full Ascend + MindSpore stack.

## 1. Repository Guardrails

- Primary repository root: `F:\ICT\义肢核心代码\_incoming\HuaWei_ict_github`
- Primary development/exec root: `F:\ICT\义肢核心代码\_incoming\HuaWei_ict_github\code`
- Use this as the only active codebase for handoff.

## 2. Environment Expectations

Minimum:

- Python >= 3.8
- `numpy`, `scipy`, `pyyaml`

Ascend target machine:

- `mindspore==2.7.1` (or target-validated equivalent)
- `mindspore_lite`
- Optional hardware runtime libs: `pyserial`, `smbus2`

## 3. Preflight

Local machine (without MindSpore):

```powershell
python scripts/preflight.py --mode local
```

Ascend machine:

```powershell
python scripts/preflight.py --mode ascend
```

If preflight reports errors, fix those first.

## 4. Local Non-MindSpore Validation

```powershell
python -m compileall .
python tests/test_gestures.py
python tests/test_preprocessing.py
python tests/test_csv_dataset.py
python tests/test_integration.py
python scripts/preflight.py --mode local
python -m runtime.run --standalone --max_cycles 200
```

Notes:

- These checks validate structure, config consistency, dataset parsing, preprocessing, and runtime loop behavior.
- They do **not** prove MindSpore training/export on this machine.

## 5. Ascend Machine Validation

Run from `F:\ICT\义肢核心代码\_incoming\HuaWei_ict_github\code`:

```powershell
python scripts/preflight.py --mode ascend
python -m training.train --data_dir ../data --config configs/training.yaml --epochs 1
python -m conversion.convert --checkpoint checkpoints/neurogrip_best.ckpt --output models/neurogrip --config configs/conversion.yaml
python scripts/preflight.py --mode ascend
python -m runtime.run --config configs/runtime.yaml --standalone --max_cycles 300
python -m runtime.run --config configs/runtime.yaml
```

## 6. Common Failure Modes

1. `mindspore`/`mindspore_lite` missing:
- Install platform-specific packages first.

2. Runtime model file missing (`models/neurogrip.mindir`):
- Run training + conversion first.

3. Runtime model build fails on Lite backend (example keywords):
- `AdaptiveAvgPool2D`, `unsupported primitive type`
- `GetPrimitiveCreator failed`, `InferSubgraph failed`, `ret = -500`
- `build_from_file failed! Error is Common error code.`
- Actions:
  - Ensure runtime device is configured as `Ascend` on target machine.
  - Re-export `.mindir` after compatibility fixes; do not rely on old model artifacts.
  - Re-run `python scripts/preflight.py --mode ascend` after conversion.

4. Shape mismatch at runtime:
- Ensure preprocess settings are aligned across:
  - `configs/training.yaml`
  - `configs/conversion.yaml`
  - `configs/runtime.yaml`

5. Hardware communication issues:
- Validate serial port / I2C settings in `configs/runtime.yaml`.
- Use `--standalone --max_cycles N` first to isolate software from hardware issues.

## 7. Definition of Done for Handoff

- Preflight passes on target machine.
- Training command runs and emits checkpoint.
- Conversion command emits `.mindir`.
- Standalone runtime loop runs and exits cleanly with `--max_cycles`.
- Real runtime command starts and controls hardware as expected.
