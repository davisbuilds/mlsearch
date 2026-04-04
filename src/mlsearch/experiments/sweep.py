from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from mlsearch.config import TrainConfig, load_train_config, merge_train_config
from mlsearch.eval.run_eval import ensure_baseline_compatible, load_latest_report, run_model_eval
from mlsearch.experiments.compare import compare_metric_sets
from mlsearch.experiments.logging import append_result
from mlsearch.paths import PATHS
from mlsearch.training.train_retriever import train_retriever


@dataclass(frozen=True)
class SweepRunResult:
    model_ref: str
    config: dict[str, object]
    metrics: dict[str, float]
    comparison: dict[str, float | str]
    reference_model_ref: str
    report_path: str
    status: str


@dataclass(frozen=True)
class SweepReport:
    report_path: str
    baseline_report_path: str
    baseline_metrics: dict[str, float]
    champion_model_ref: str
    champion_metrics: dict[str, float]
    run_count: int
    runs: list[SweepRunResult]


def run_experiment_sweep(
    *,
    config_path: Path,
    reference_model: str,
    learning_rates: list[float] | None = None,
    num_epochs: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    max_examples: list[int] | None = None,
    seeds: list[int] | None = None,
    record_results: bool,
) -> SweepReport:
    base_config = load_train_config(config_path)
    variant_configs = build_sweep_variants(
        base_config=base_config,
        learning_rates=learning_rates,
        num_epochs=num_epochs,
        batch_sizes=batch_sizes,
        max_examples=max_examples,
        seeds=seeds,
    )

    baseline_report_path = _latest_baseline_report_path()
    baseline_report = load_latest_report("baseline")
    baseline_candidates_path = Path(str(baseline_report["candidates_path"]))
    ensure_baseline_compatible(
        baseline_report,
        candidates_path=baseline_candidates_path,
    )
    baseline_metrics = dict(baseline_report["metrics"])
    champion_model_ref, champion_metrics = _resolve_reference_metrics(
        reference_model=reference_model,
        baseline_candidates_path=baseline_candidates_path,
        baseline_metrics=baseline_metrics,
    )
    run_results: list[SweepRunResult] = []

    for variant in variant_configs:
        train_report = train_retriever(config=variant)
        eval_report = run_model_eval(model_ref=Path(train_report.model_dir))
        comparison = compare_metric_sets(eval_report.metrics, champion_metrics)
        status = str(comparison["status"])
        model_ref = Path(train_report.model_dir).name
        if record_results:
            append_result(
                PATHS.root / "results.tsv",
                model_ref=model_ref,
                metrics=eval_report.metrics,
                status=status,
                description=_render_description(variant, champion_model_ref),
            )
        run_results.append(
            SweepRunResult(
                model_ref=model_ref,
                config=asdict(variant),
                metrics=eval_report.metrics,
                comparison=comparison,
                reference_model_ref=champion_model_ref,
                report_path=eval_report.report_path,
                status=status,
            )
        )
        if status == "keep":
            champion_metrics = eval_report.metrics
            champion_model_ref = model_ref

    report_path = _write_sweep_report(
        {
            "baseline_report_path": baseline_report_path,
            "baseline_candidates_path": str(baseline_candidates_path),
            "baseline_metrics": baseline_metrics,
            "initial_reference_model_ref": champion_model_ref,
            "champion_model_ref": champion_model_ref,
            "champion_metrics": champion_metrics,
            "run_count": len(run_results),
            "runs": [
                {
                    "model_ref": run.model_ref,
                    "config": run.config,
                    "metrics": run.metrics,
                    "comparison": run.comparison,
                    "reference_model_ref": run.reference_model_ref,
                    "report_path": run.report_path,
                    "status": run.status,
                }
                for run in run_results
            ],
        }
    )
    return SweepReport(
        report_path=str(report_path),
        baseline_report_path=baseline_report_path,
        baseline_metrics=baseline_metrics,
        champion_model_ref=champion_model_ref,
        champion_metrics=champion_metrics,
        run_count=len(run_results),
        runs=run_results,
    )


def build_sweep_variants(
    *,
    base_config: TrainConfig,
    learning_rates: list[float] | None = None,
    num_epochs: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    max_examples: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[TrainConfig]:
    values_by_field: list[tuple[str, list[object]]] = [
        ("learning_rate", [*dict.fromkeys(learning_rates or [base_config.learning_rate])]),
        ("num_epochs", [*dict.fromkeys(num_epochs or [base_config.num_epochs])]),
        ("batch_size", [*dict.fromkeys(batch_sizes or [base_config.batch_size])]),
        ("max_examples", [*dict.fromkeys(max_examples or [base_config.max_examples])]),
        ("seed", [*dict.fromkeys(seeds or [base_config.seed])]),
    ]
    fields = [name for name, _ in values_by_field]
    value_lists = [values for _, values in values_by_field]

    variants: list[TrainConfig] = []
    for values in product(*value_lists):
        overrides = dict(zip(fields, values, strict=True))
        variants.append(merge_train_config(base_config, **overrides))
    return variants


def _latest_baseline_report_path() -> str:
    reports = sorted(PATHS.artifacts_results.glob("baseline-*.json"))
    if not reports:
        raise FileNotFoundError(f"No baseline reports found in {PATHS.artifacts_results}")
    return str(reports[-1])


def _resolve_reference_metrics(
    *,
    reference_model: str,
    baseline_candidates_path: Path,
    baseline_metrics: dict[str, float],
) -> tuple[str, dict[str, float]]:
    if reference_model == "baseline":
        return "baseline", baseline_metrics
    reference_report = run_model_eval(model_ref=reference_model)
    if Path(reference_report.candidates_path) != baseline_candidates_path:
        raise ValueError(
            "Reference model targets a different benchmark split; rerun `eval baseline` before starting the sweep."
        )
    return reference_report.model_ref, reference_report.metrics


def _render_description(config: TrainConfig, reference_model_ref: str) -> str:
    return (
        "experiment sweep run "
        f"vs={reference_model_ref} "
        f"lr={config.learning_rate:g} "
        f"epochs={config.num_epochs} "
        f"batch={config.batch_size} "
        f"max_examples={config.max_examples} "
        f"seed={config.seed}"
    )


def _write_sweep_report(payload: dict[str, object]) -> Path:
    PATHS.artifacts_results.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = PATHS.artifacts_results / f"sweep-{timestamp}.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path
