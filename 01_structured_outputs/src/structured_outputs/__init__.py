"""Extraction d'informations structurées sur un candidat."""

from structured_outputs.benchmark import run_benchmark
from structured_outputs.evaluation import evaluate_dataset, evaluate_profile
from structured_outputs.extractor import (
    CandidateExtractor,
    ExtractionError,
    ExtractionResult,
)
from structured_outputs.models import CandidateProfile

__all__ = [
    "CandidateExtractor",
    "CandidateProfile",
    "ExtractionError",
    "ExtractionResult",
    "evaluate_dataset",
    "evaluate_profile",
    "run_benchmark",
]
