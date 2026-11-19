"""Controlled one-factor and combined-grid experiment runner."""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MATPLOTLIB_CONFIG = Path(__file__).resolve().parents[1] / "data/cache/matplotlib"
_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Settings
from src.evaluation import evaluate_questions
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete retrieval/index configuration."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str

    def key(self) -> tuple[int, int, int, str]:
        return (self.chunk_size, self.chunk_overlap, self.top_k, self.embedding_model)


def make_experiment_configs(include_combined_grid: bool = True) -> list[tuple[str, ExperimentConfig]]:
    """Create a baseline, one-variable-at-a-time set, then an optional full grid."""
    baseline = ExperimentConfig(1024, 0, 2, EMBEDDING_MODELS[0])
    result: list[tuple[str, ExperimentConfig]] = [("baseline", baseline)]
    one_factor = [
        ("chunk_size", ExperimentConfig(value, baseline.chunk_overlap, baseline.top_k, baseline.embedding_model))
        for value in (256, 512, 768, 1024)
    ]
    one_factor += [
        ("chunk_overlap", ExperimentConfig(baseline.chunk_size, value, baseline.top_k, baseline.embedding_model))
        for value in (0, 50, 100)
    ]
    one_factor += [
        ("top_k", ExperimentConfig(baseline.chunk_size, baseline.chunk_overlap, value, baseline.embedding_model))
        for value in (2, 3, 5, 8)
    ]
    one_factor += [
        ("embedding_model", ExperimentConfig(baseline.chunk_size, baseline.chunk_overlap, baseline.top_k, value))
        for value in EMBEDDING_MODELS
    ]
    for variable, config in one_factor:
        if config.key() != baseline.key():
            result.append((f"one_factor:{variable}", config))

    if include_combined_grid:
        seen = {config.key() for _, config in result}
        for values in itertools.product((256, 512, 768, 1024), (0, 50, 100), (2, 3, 5, 8), EMBEDDING_MODELS):
            config = ExperimentConfig(*values)
            if config.key() not in seen:
                result.append(("combined_grid", config))
                seen.add(config.key())
    return result


def _save_plots(frame: pd.DataFrame, results_dir: Path) -> None:
    """Write accuracy plots using actual observations only."""
    metric = "llm_judge_accuracy" if frame["llm_judge_accuracy"].notna().any() else "heuristic_accuracy"
    charts = (
        ("chunk_size", "Chunk size", "accuracy_vs_chunk_size.png"),
        ("top_k", "Top-k", "accuracy_vs_top_k.png"),
        ("embedding_model", "Embedding model", "accuracy_vs_embedding_model.png"),
    )
    for column, label, filename in charts:
        group = frame.groupby(column, dropna=False)[metric].agg(["mean", "std"]).reset_index()
        fig, axis = plt.subplots(figsize=(8, 4.5))
        if column == "embedding_model":
            x_values = np.arange(len(group))
            axis.bar(x_values, group["mean"], yerr=group["std"].fillna(0), capsize=3)
            axis.set_xticks(x_values, group[column], rotation=20, ha="right")
        else:
            axis.errorbar(group[column], group["mean"], yerr=group["std"].fillna(0), marker="o", capsize=3)
        axis.set_xlabel(label)
        axis.set_ylabel(f"Accuracy ({metric})")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=150)
        plt.close(fig)


def _write_summary(frame: pd.DataFrame, results_dir: Path) -> None:
    """Write a source-backed baseline/best comparison to Markdown."""
    metric = "llm_judge_accuracy" if frame["llm_judge_accuracy"].notna().any() else "heuristic_accuracy"
    baseline_rows = frame[frame["experiment"] == "baseline"]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else None
    best = frame.sort_values([metric, "latency_mean_s"], ascending=[False, True]).iloc[0]
    lines = [
        "# Experiment summary",
        "",
        f"Accuracy metric used for ranking: `{metric}`.",
        f"Measured configurations: {len(frame)}.",
        "",
        "| Result | Accuracy | Mean latency | p95 latency | Configuration |",
        "|---|---:|---:|---:|---|",
    ]
    for label, row in (("Baseline", baseline), ("Best measured", best)):
        if row is None:
            continue
        conf = (
            f"chunk_size={int(row['chunk_size'])}, overlap={int(row['chunk_overlap'])}, "
            f"top_k={int(row['top_k'])}, embedding={row['embedding_model']}"
        )
        lines.append(
            f"| {label} | {row[metric]:.4f} | {row['latency_mean_s']:.3f}s | "
            f"{row['latency_p95_s']:.3f}s | {conf} |"
        )
    if baseline is not None:
        lines.extend(
            [
                "",
                f"Best minus baseline accuracy: {float(best[metric] - baseline[metric]):+.4f}.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Latency covers the pipeline's retrieval and answer generation for each question. "
            "The heuristic score requires the labeled source page to be retrieved and at least "
            "35% of ground-truth terms to occur in the answer. It is not an LLM quality judgment.",
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiments(
    settings: Settings,
    questions: list[dict[str, Any]],
    *,
    include_combined_grid: bool = True,
) -> pd.DataFrame:
    """Run evaluation across controlled and combined configurations."""
    if len(questions) != 50:
        raise ValueError(
            f"Expected 50 verified benchmark questions; got {len(questions)}. "
            "Refusing to produce partial or fabricated experiment results."
        )
    results_dir = settings.path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    question_frames: list[pd.DataFrame] = []
    for ordinal, (kind, config) in enumerate(make_experiment_configs(include_combined_grid), start=1):
        logger.info(
            "Experiment %d/%d (%s): chunk=%d overlap=%d top_k=%d embedding=%s",
            ordinal,
            len(make_experiment_configs(include_combined_grid)),
            kind,
            config.chunk_size,
            config.chunk_overlap,
            config.top_k,
            config.embedding_model,
        )
        experiment_settings = settings.with_overrides(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            top_k=config.top_k,
            embedding_model=config.embedding_model,
        )
        pipeline = RAGPipeline(settings=experiment_settings)
        pipeline.build_index()
        metrics, question_rows = evaluate_questions(pipeline, questions)
        conf_id = f"{ordinal:03d}"
        summary_rows.append(
            {
                "config_id": conf_id,
                "experiment": "baseline" if kind == "baseline" else kind,
                **config.__dict__,
                **metrics,
            }
        )
        for row in question_rows:
            row["config_id"] = conf_id
        question_frames.append(pd.DataFrame(question_rows))

    frame = pd.DataFrame(summary_rows)
    frame.to_csv(results_dir / "experiments.csv", index=False)
    pd.concat(question_frames, ignore_index=True).to_csv(results_dir / "per_question.csv", index=False)
    _save_plots(frame, results_dir)
    _write_summary(frame, results_dir)
    return frame

