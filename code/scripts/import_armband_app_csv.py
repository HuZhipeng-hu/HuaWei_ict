"""Import armband app CSV exports into the standardized dataset layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.collection_utils import (
    build_manifest_row,
    build_quality_console_line,
    build_relative_recording_path,
    ensure_unique_path,
    evaluate_recording_quality,
    gather_source_csvs,
    load_collection_protocol,
    normalize_relative_path,
    read_source_csv,
    resolve_manifest_path,
    resolve_report_dir,
    timestamp_from_path,
    upsert_recordings_manifest,
    validate_metadata,
    write_json_report,
    write_standard_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import armband app CSV exports into standardized dataset files")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gesture", required=True)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--wearing_state", required=True)
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--source_csv", action="append", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Keep source CSVs after import (default).")
    mode.add_argument("--move", action="store_true", help="Delete source CSVs after successful import.")
    return parser


def run_import_batch(
    *,
    data_dir: str | Path,
    gesture: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    training_config: str | Path = "configs/training.yaml",
    manifest_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    source_csvs: Sequence[str | Path] | None = None,
    move: bool = False,
) -> dict[str, Any]:
    metadata = validate_metadata(
        gesture=gesture,
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        wearing_state=wearing_state,
    )
    preprocess_cfg, quality_filter = load_collection_protocol(training_config)
    sources = gather_source_csvs(source_dir=source_dir, source_csvs=source_csvs)

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_file = resolve_manifest_path(data_root, manifest_path)
    report_root = resolve_report_dir(data_root, report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    print("Training currently consumes all 8 EMG channels under the 16x24x6 dual-branch protocol.")

    records: list[dict[str, Any]] = []
    last_timestamp = None
    for index, source_path in enumerate(sources, start=1):
        matrix = read_source_csv(source_path)
        timestamp = timestamp_from_path(source_path)
        last_timestamp = timestamp
        relative_path = build_relative_recording_path(
            metadata,
            timestamp=timestamp,
            recording_index=index,
        )
        destination = ensure_unique_path(data_root / relative_path)
        write_standard_csv(destination, matrix)

        quality_report = evaluate_recording_quality(
            matrix,
            preprocess_config=preprocess_cfg,
            quality_filter=quality_filter,
        )
        rel_for_manifest = normalize_relative_path(destination.relative_to(data_root))
        manifest_row = build_manifest_row(
            relative_path=rel_for_manifest,
            metadata=metadata,
            timestamp=timestamp,
            sample_count=int(matrix.shape[0]),
            quality_report=quality_report,
            source_origin="armband_app_import",
        )
        upsert_recordings_manifest(manifest_file, manifest_row)

        if move:
            source_path.unlink()

        record = {
            "source_path": str(source_path.resolve()),
            "absolute_path": str(destination.resolve()),
            "relative_path": rel_for_manifest,
            "gesture": metadata.gesture,
            "timestamp": timestamp,
            "source_origin": "armband_app_import",
            **quality_report,
        }
        records.append(record)
        print(f"[import {index}/{len(sources)}] {destination.name}: {build_quality_console_line(record)}")

    report_stamp = last_timestamp or "import"
    report_path = report_root / f"{report_stamp}_{metadata.gesture.lower()}_{metadata.session_id}_import.json"
    write_json_report(
        report_path,
        {
            "mode": "import",
            "training_config": str(training_config),
            "manifest_path": str(Path(manifest_file).resolve()),
            "data_dir": str(data_root.resolve()),
            "records": records,
        },
    )

    return {
        "manifest_path": str(Path(manifest_file).resolve()),
        "report_path": str(Path(report_path).resolve()),
        "records": records,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_import_batch(
        data_dir=args.data_dir,
        gesture=args.gesture,
        user_id=args.user_id,
        session_id=args.session_id,
        device_id=args.device_id,
        wearing_state=args.wearing_state,
        training_config=args.training_config,
        manifest_path=args.manifest_path,
        report_dir=args.report_dir,
        source_dir=args.source_dir,
        source_csvs=args.source_csv,
        move=bool(args.move),
    )
    print(f"Saved report: {result['report_path']}")
    print(f"Updated manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
