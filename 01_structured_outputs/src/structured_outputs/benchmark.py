"""Orchestration séquentielle d'extractions et de leur évaluation."""

from collections.abc import Iterable
from dataclasses import dataclass

from structured_outputs.dataset import EvaluationExample
from structured_outputs.evaluation import (
    DatasetEvaluation,
    ProfileEvaluation,
    evaluate_dataset,
    evaluate_profile,
)
from structured_outputs.extractor import CandidateExtractor
from structured_outputs.models import CandidateProfile


@dataclass(frozen=True)
class BenchmarkExampleResult:
    id: str
    expected: CandidateProfile
    predicted: CandidateProfile
    evaluation: ProfileEvaluation
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    model: str


@dataclass(frozen=True)
class BenchmarkResult:
    examples: list[BenchmarkExampleResult]
    evaluation: DatasetEvaluation
    total_latency_seconds: float
    mean_latency_seconds: float
    total_input_tokens: int | None
    total_output_tokens: int | None
    model: str | tuple[str, ...]


def _sum_known_tokens(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def run_benchmark(
    examples: Iterable[EvaluationExample],
    extractor: CandidateExtractor,
) -> BenchmarkResult:
    """Extraire et scorer dans l'ordre, en propageant immédiatement les erreurs.

    Les latences sont celles de l'extractor. Les totaux de tokens ne comprennent
    que les valeurs connues, ou None si toutes sont absentes. Le modèle est une
    chaîne si unique, sinon un tuple de modèles distincts dans l'ordre rencontré.
    Aucun client n'est construit ici ; l'appelant choisit et injecte l'extractor.
    """
    results: list[BenchmarkExampleResult] = []
    for example in examples:
        extraction = extractor.extract(example.text)
        results.append(
            BenchmarkExampleResult(
                id=example.id,
                expected=example.expected,
                predicted=extraction.profile,
                evaluation=evaluate_profile(extraction.profile, example.expected),
                latency_seconds=extraction.latency_seconds,
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                model=extraction.model,
            )
        )
    if not results:
        raise ValueError("Un benchmark nécessite au moins un exemple.")

    evaluation = evaluate_dataset((item.predicted, item.expected) for item in results)
    total_latency = sum(item.latency_seconds for item in results)
    models = tuple(dict.fromkeys(item.model for item in results))
    return BenchmarkResult(
        examples=results,
        evaluation=evaluation,
        total_latency_seconds=total_latency,
        mean_latency_seconds=total_latency / len(results),
        total_input_tokens=_sum_known_tokens(item.input_tokens for item in results),
        total_output_tokens=_sum_known_tokens(item.output_tokens for item in results),
        model=models[0] if len(models) == 1 else models,
    )
