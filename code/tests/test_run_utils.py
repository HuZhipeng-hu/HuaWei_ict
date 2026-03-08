from pathlib import Path

from shared.run_utils import append_csv_row, build_run_id, ensure_run_dir


def test_build_run_id_includes_tag_suffix():
    run_id = build_run_id("baseline")
    assert run_id.endswith("_baseline")
    assert len(run_id.split("_")) >= 3


def test_ensure_run_dir_and_append_csv_row(tmp_path: Path):
    run_id, run_dir = ensure_run_dir(tmp_path, None, default_tag="train")
    assert run_dir.exists()
    assert run_dir.name == run_id

    csv_path = tmp_path / "results.csv"
    append_csv_row(csv_path, ["run_id", "score"], {"run_id": run_id, "score": 0.95})
    append_csv_row(csv_path, ["run_id", "score"], {"run_id": "next", "score": 0.96})

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "run_id,score"
    assert len(lines) == 3
