"""Extraction d'informations structurées sur un candidat."""

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
]
