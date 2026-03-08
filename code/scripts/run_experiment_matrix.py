"""Generate command matrix for 24 structured experiments."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured experiment commands")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--manifest", default="artifacts/splits/manifest_v2.json")
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", default="logs/experiment_matrix_commands.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    commands = []

    # Stage A (6)
    for seed in [11, 22, 33]:
        commands.append(
            f'{args.python} -m training.train --config {args.config} --data_dir {args.data_dir} '
            f"--split_manifest_in {args.manifest} --manifest_strategy v2 --quality_report_out logs/quality/a_seed_{seed}.json"
        )
    for enabled, seed in [(True, 44), (False, 55), (True, 66)]:
        commands.append(
            f'{args.python} -m training.train --config {args.config} --data_dir {args.data_dir} '
            f"--split_manifest_in {args.manifest} --manifest_strategy v2 "
            f"--quality_report_out logs/quality/a_filter_{int(enabled)}_{seed}.json"
        )

    # Stage B (8)
    loss_types = ["ce", "focal", "cb_focal", "cb_focal"]
    ema_flags = [False, False, False, True]
    for (loss, ema), seed in zip(zip(loss_types, ema_flags), [101, 102, 201, 202, 301, 302, 401, 402]):
        commands.append(
            f"# set in config: training.loss.type={loss}, training.ema.enabled={str(ema).lower()}, seed={seed}"
        )
        commands.append(
            f'{args.python} -m training.train --config {args.config} --data_dir {args.data_dir} '
            f"--split_manifest_in {args.manifest} --manifest_strategy v2"
        )

    # Stage C (6)
    for hard_ratio, tta in itertools.product([0.0, 0.3, 0.5], ["off", "on"]):
        commands.append(
            f"# set in config: training.sampler.hard_mining_ratio={hard_ratio}, "
            f"runtime.inference.tta_offsets={'[0.0]' if tta=='off' else '[0.0,0.33,0.66]'}"
        )
        commands.append(
            f'{args.python} -m training.train --config {args.config} --data_dir {args.data_dir} '
            f"--split_manifest_in {args.manifest} --manifest_strategy v2"
        )

    # Stage D (4)
    for idx in [1, 2]:
        for seed in [901 + idx, 951 + idx]:
            commands.append(f"# rerun top-{idx} setup with seed={seed}")
            commands.append(
                f'{args.python} -m training.train --config {args.config} --data_dir {args.data_dir} '
                f"--split_manifest_in {args.manifest} --manifest_strategy v2"
            )

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} lines to {out}")


if __name__ == "__main__":
    main()

