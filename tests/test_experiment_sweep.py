from __future__ import annotations

from mlsearch.config import TrainConfig
from mlsearch.experiments.sweep import build_sweep_variants


def test_build_sweep_variants_expands_cartesian_product() -> None:
    variants = build_sweep_variants(
        base_config=TrainConfig(device="mps", batch_size=4, max_examples=2000),
        learning_rates=[1e-5, 2e-5],
        num_epochs=[1, 2],
        batch_sizes=[4],
        max_examples=[1000],
        seeds=[42],
    )

    assert len(variants) == 4
    assert {variant.learning_rate for variant in variants} == {1e-5, 2e-5}
    assert {variant.num_epochs for variant in variants} == {1, 2}
    assert {variant.batch_size for variant in variants} == {4}
    assert {variant.max_examples for variant in variants} == {1000}
    assert all(variant.device == "mps" for variant in variants)


def test_build_sweep_variants_deduplicates_duplicate_values() -> None:
    variants = build_sweep_variants(
        base_config=TrainConfig(),
        learning_rates=[2e-5, 2e-5],
        num_epochs=[1, 1],
        batch_sizes=[8, 8],
        max_examples=[5000, 5000],
        seeds=[42, 42],
    )

    assert len(variants) == 1
    assert variants[0] == TrainConfig()
