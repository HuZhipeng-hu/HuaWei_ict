# NeuroGrip Experiment Matrix

- Training config: `configs/training.yaml`
- Data dir: `../data`
- Locked manifest: `artifacts/splits/default_split_manifest.json`
- Run root: `artifacts/runs`

## Aggregate CSV Headers

```text
run_id,manifest_path,checkpoint_path,model_type,base_channels,use_se,loss_type,hard_mining_ratio,augment_enabled,augment_factor,use_mixup,test_accuracy,test_macro_f1,test_macro_recall,top_confusion_pair,hit_rate,false_trigger_rate,latency_p50_ms,latency_p95_ms
```

## Stage A / a_standard_b16_se

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true'}`
- Expected run dir: `artifacts/runs/a_standard_b16_se`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_se --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b16_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_se
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b16_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_se
```

## Stage A / a_standard_b16_nose

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_standard_b16_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_nose --manifest_strategy v2 --model_type standard --base_channels 16 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b16_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b16_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b16_nose
```

## Stage A / a_standard_b24_se

- Overrides: `{'model_type': 'standard', 'base_channels': 24, 'use_se': 'true'}`
- Expected run dir: `artifacts/runs/a_standard_b24_se`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_se --manifest_strategy v2 --model_type standard --base_channels 24 --use_se true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b24_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_se
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b24_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_se
```

## Stage A / a_standard_b24_nose

- Overrides: `{'model_type': 'standard', 'base_channels': 24, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_standard_b24_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_nose --manifest_strategy v2 --model_type standard --base_channels 24 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b24_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b24_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b24_nose
```

## Stage A / a_standard_b32_se

- Overrides: `{'model_type': 'standard', 'base_channels': 32, 'use_se': 'true'}`
- Expected run dir: `artifacts/runs/a_standard_b32_se`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_se --manifest_strategy v2 --model_type standard --base_channels 32 --use_se true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b32_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_se
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b32_se/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_se
```

## Stage A / a_standard_b32_nose

- Overrides: `{'model_type': 'standard', 'base_channels': 32, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_standard_b32_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_nose --manifest_strategy v2 --model_type standard --base_channels 32 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b32_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_standard_b32_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_standard_b32_nose
```

## Stage A / a_lite_b16_nose

- Overrides: `{'model_type': 'lite', 'base_channels': 16, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_lite_b16_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b16_nose --manifest_strategy v2 --model_type lite --base_channels 16 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b16_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b16_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b16_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b16_nose
```

## Stage A / a_lite_b24_nose

- Overrides: `{'model_type': 'lite', 'base_channels': 24, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_lite_b24_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b24_nose --manifest_strategy v2 --model_type lite --base_channels 24 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b24_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b24_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b24_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b24_nose
```

## Stage A / a_lite_b32_nose

- Overrides: `{'model_type': 'lite', 'base_channels': 32, 'use_se': 'false'}`
- Expected run dir: `artifacts/runs/a_lite_b32_nose`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b32_nose --manifest_strategy v2 --model_type lite --base_channels 32 --use_se false
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b32_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b32_nose
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/a_lite_b32_nose/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id a_lite_b32_nose
```

## Stage B/C / bc_01_focal_hm00_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.0, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_01_focal_hm00_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_01_focal_hm00_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.0 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_01_focal_hm00_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_01_focal_hm00_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_01_focal_hm00_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_01_focal_hm00_a2_nomix
```

## Stage B/C / bc_02_focal_hm00_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.0, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_02_focal_hm00_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_02_focal_hm00_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.0 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_02_focal_hm00_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_02_focal_hm00_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_02_focal_hm00_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_02_focal_hm00_a2_mix
```

## Stage B/C / bc_03_focal_hm00_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.0, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_03_focal_hm00_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_03_focal_hm00_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.0 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_03_focal_hm00_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_03_focal_hm00_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_03_focal_hm00_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_03_focal_hm00_a3_nomix
```

## Stage B/C / bc_04_focal_hm00_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.0, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_04_focal_hm00_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_04_focal_hm00_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.0 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_04_focal_hm00_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_04_focal_hm00_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_04_focal_hm00_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_04_focal_hm00_a3_mix
```

## Stage B/C / bc_05_focal_hm03_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.3, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_05_focal_hm03_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_05_focal_hm03_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.3 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_05_focal_hm03_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_05_focal_hm03_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_05_focal_hm03_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_05_focal_hm03_a2_nomix
```

## Stage B/C / bc_06_focal_hm03_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.3, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_06_focal_hm03_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_06_focal_hm03_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.3 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_06_focal_hm03_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_06_focal_hm03_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_06_focal_hm03_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_06_focal_hm03_a2_mix
```

## Stage B/C / bc_07_focal_hm03_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.3, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_07_focal_hm03_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_07_focal_hm03_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.3 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_07_focal_hm03_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_07_focal_hm03_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_07_focal_hm03_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_07_focal_hm03_a3_nomix
```

## Stage B/C / bc_08_focal_hm03_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.3, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_08_focal_hm03_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_08_focal_hm03_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.3 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_08_focal_hm03_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_08_focal_hm03_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_08_focal_hm03_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_08_focal_hm03_a3_mix
```

## Stage B/C / bc_09_focal_hm05_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.5, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_09_focal_hm05_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_09_focal_hm05_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.5 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_09_focal_hm05_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_09_focal_hm05_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_09_focal_hm05_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_09_focal_hm05_a2_nomix
```

## Stage B/C / bc_10_focal_hm05_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.5, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_10_focal_hm05_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_10_focal_hm05_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.5 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_10_focal_hm05_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_10_focal_hm05_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_10_focal_hm05_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_10_focal_hm05_a2_mix
```

## Stage B/C / bc_11_focal_hm05_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.5, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_11_focal_hm05_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_11_focal_hm05_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.5 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_11_focal_hm05_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_11_focal_hm05_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_11_focal_hm05_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_11_focal_hm05_a3_nomix
```

## Stage B/C / bc_12_focal_hm05_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'focal', 'hard_mining_ratio': 0.5, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_12_focal_hm05_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_12_focal_hm05_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type focal --hard_mining_ratio 0.5 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_12_focal_hm05_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_12_focal_hm05_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_12_focal_hm05_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_12_focal_hm05_a3_mix
```

## Stage B/C / bc_13_cb_focal_hm00_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.0, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_13_cb_focal_hm00_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_13_cb_focal_hm00_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.0 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_13_cb_focal_hm00_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_13_cb_focal_hm00_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_13_cb_focal_hm00_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_13_cb_focal_hm00_a2_nomix
```

## Stage B/C / bc_14_cb_focal_hm00_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.0, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_14_cb_focal_hm00_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_14_cb_focal_hm00_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.0 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_14_cb_focal_hm00_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_14_cb_focal_hm00_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_14_cb_focal_hm00_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_14_cb_focal_hm00_a2_mix
```

## Stage B/C / bc_15_cb_focal_hm00_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.0, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_15_cb_focal_hm00_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_15_cb_focal_hm00_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.0 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_15_cb_focal_hm00_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_15_cb_focal_hm00_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_15_cb_focal_hm00_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_15_cb_focal_hm00_a3_nomix
```

## Stage B/C / bc_16_cb_focal_hm00_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.0, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_16_cb_focal_hm00_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_16_cb_focal_hm00_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.0 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_16_cb_focal_hm00_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_16_cb_focal_hm00_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_16_cb_focal_hm00_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_16_cb_focal_hm00_a3_mix
```

## Stage B/C / bc_17_cb_focal_hm03_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.3, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_17_cb_focal_hm03_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_17_cb_focal_hm03_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.3 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_17_cb_focal_hm03_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_17_cb_focal_hm03_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_17_cb_focal_hm03_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_17_cb_focal_hm03_a2_nomix
```

## Stage B/C / bc_18_cb_focal_hm03_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.3, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_18_cb_focal_hm03_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_18_cb_focal_hm03_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.3 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_18_cb_focal_hm03_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_18_cb_focal_hm03_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_18_cb_focal_hm03_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_18_cb_focal_hm03_a2_mix
```

## Stage B/C / bc_19_cb_focal_hm03_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.3, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_19_cb_focal_hm03_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_19_cb_focal_hm03_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.3 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_19_cb_focal_hm03_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_19_cb_focal_hm03_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_19_cb_focal_hm03_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_19_cb_focal_hm03_a3_nomix
```

## Stage B/C / bc_20_cb_focal_hm03_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.3, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_20_cb_focal_hm03_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_20_cb_focal_hm03_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.3 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_20_cb_focal_hm03_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_20_cb_focal_hm03_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_20_cb_focal_hm03_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_20_cb_focal_hm03_a3_mix
```

## Stage B/C / bc_21_cb_focal_hm05_a2_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.5, 'augment_factor': 2, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_21_cb_focal_hm05_a2_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_21_cb_focal_hm05_a2_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.5 --augment_factor 2 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_21_cb_focal_hm05_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_21_cb_focal_hm05_a2_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_21_cb_focal_hm05_a2_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_21_cb_focal_hm05_a2_nomix
```

## Stage B/C / bc_22_cb_focal_hm05_a2_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.5, 'augment_factor': 2, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_22_cb_focal_hm05_a2_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_22_cb_focal_hm05_a2_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.5 --augment_factor 2 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_22_cb_focal_hm05_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_22_cb_focal_hm05_a2_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_22_cb_focal_hm05_a2_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_22_cb_focal_hm05_a2_mix
```

## Stage B/C / bc_23_cb_focal_hm05_a3_nomix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.5, 'augment_factor': 3, 'use_mixup': 'false', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_23_cb_focal_hm05_a3_nomix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_23_cb_focal_hm05_a3_nomix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.5 --augment_factor 3 --use_mixup false --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_23_cb_focal_hm05_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_23_cb_focal_hm05_a3_nomix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_23_cb_focal_hm05_a3_nomix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_23_cb_focal_hm05_a3_nomix
```

## Stage B/C / bc_24_cb_focal_hm05_a3_mix

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'loss_type': 'cb_focal', 'hard_mining_ratio': 0.5, 'augment_factor': 3, 'use_mixup': 'true', 'augmentation_enabled': 'true'}`
- Expected run dir: `artifacts/runs/bc_24_cb_focal_hm05_a3_mix`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_24_cb_focal_hm05_a3_mix --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --loss_type cb_focal --hard_mining_ratio 0.5 --augment_factor 3 --use_mixup true --augmentation_enabled true
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/bc_24_cb_focal_hm05_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_24_cb_focal_hm05_a3_mix
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/bc_24_cb_focal_hm05_a3_mix/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id bc_24_cb_focal_hm05_a3_mix
```

## Stage D / d_top1_seed901

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 901}`
- Expected run dir: `artifacts/runs/d_top1_seed901`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed901 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 901
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed901/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed901
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed901/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed901
```

## Stage D / d_top1_seed902

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 902}`
- Expected run dir: `artifacts/runs/d_top1_seed902`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed902 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 902
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed902/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed902
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed902/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed902
```

## Stage D / d_top1_seed951

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 951}`
- Expected run dir: `artifacts/runs/d_top1_seed951`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed951 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 951
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed951/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed951
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed951/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed951
```

## Stage D / d_top1_seed952

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 952}`
- Expected run dir: `artifacts/runs/d_top1_seed952`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed952 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 952
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed952/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed952
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top1_seed952/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top1_seed952
```

## Stage D / d_top2_seed901

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 901}`
- Expected run dir: `artifacts/runs/d_top2_seed901`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed901 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 901
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed901/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed901
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed901/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed901
```

## Stage D / d_top2_seed902

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 902}`
- Expected run dir: `artifacts/runs/d_top2_seed902`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed902 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 902
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed902/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed902
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed902/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed902
```

## Stage D / d_top2_seed951

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 951}`
- Expected run dir: `artifacts/runs/d_top2_seed951`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed951 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 951
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed951/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed951
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed951/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed951
```

## Stage D / d_top2_seed952

- Overrides: `{'model_type': 'standard', 'base_channels': 16, 'use_se': 'true', 'split_seed': 952}`
- Expected run dir: `artifacts/runs/d_top2_seed952`

```bash
python -m training.train --config configs/training.yaml --data_dir ../data --split_manifest_in artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed952 --manifest_strategy v2 --model_type standard --base_channels 16 --use_se true --split_seed 952
python scripts/evaluate_ckpt.py --config configs/training.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed952/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed952
python scripts/benchmark_realtime_ckpt.py --training_config configs/training.yaml --runtime_config configs/runtime.yaml --data_dir ../data --checkpoint artifacts/runs/d_top2_seed952/checkpoints/neurogrip_best.ckpt --split_manifest artifacts/splits/default_split_manifest.json --run_root artifacts/runs --run_id d_top2_seed952
```

