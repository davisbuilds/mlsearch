from __future__ import annotations

from pathlib import Path

from mlsearch.pipelines.archive_review_artifacts import archive_review_artifacts


def test_archive_review_artifacts_copies_existing_review_files(tmp_path: Path) -> None:
    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    (reviewed_dir / "review_sample.csv").write_text("query_id\npaper-1\n", encoding="utf-8")
    (reviewed_dir / "held_out_eval.jsonl").write_text('{"query_id":"paper-1"}\n', encoding="utf-8")
    (reviewed_dir / "held_out_eval_manifest.json").write_text('{"accepted_count":1}\n', encoding="utf-8")

    report = archive_review_artifacts(reviewed_dir=reviewed_dir, label="pre-hardening")

    archive_dir = Path(report.archive_dir)
    assert archive_dir.name == "pre-hardening"
    assert (archive_dir / "review_sample.csv").exists()
    assert (archive_dir / "held_out_eval.jsonl").exists()
    assert (archive_dir / "held_out_eval_manifest.json").exists()
