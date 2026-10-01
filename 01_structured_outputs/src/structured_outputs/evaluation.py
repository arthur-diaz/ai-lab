"""Scoring déterministe de profils validés, sans filesystem ni appel LLM."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from structured_outputs.models import CandidateProfile

SCALAR_FIELDS = (
    "name",
    "job_title",
    "years_of_experience",
    "location",
    "employment_type",
    "target_salary_eur",
)


@dataclass(frozen=True)
class ProfileEvaluation:
    field_matches: dict[str, bool]
    skills_precision: float
    skills_recall: float
    skills_f1: float
    skills_exact_match: bool
    profile_exact_match: bool


@dataclass(frozen=True)
class DatasetEvaluation:
    example_count: int
    field_accuracy: dict[str, float]
    overall_field_accuracy: float
    skills_macro_precision: float
    skills_macro_recall: float
    skills_macro_f1: float
    skills_exact_match_accuracy: float
    profile_exact_match_accuracy: float


def _validate_tolerance(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} doit être une tolérance finie et positive ou nulle.")


def _numeric_match(
    predicted: float | int | None,
    expected: float | int | None,
    tolerance: float | int,
) -> bool:
    if predicted is None or expected is None:
        return predicted is expected
    if predicted == expected:
        return True
    return abs(predicted - expected) <= tolerance


def evaluate_profile(
    predicted: CandidateProfile,
    expected: CandidateProfile,
    *,
    years_tolerance: float = 0.0,
    salary_tolerance: int = 0,
) -> ProfileEvaluation:
    """Comparer les scalaires et les ensembles de skills, sans renormaliser."""
    _validate_tolerance(years_tolerance, "years_tolerance")
    _validate_tolerance(salary_tolerance, "salary_tolerance")
    matches = {
        field: getattr(predicted, field) == getattr(expected, field)
        for field in SCALAR_FIELDS
    }
    matches["years_of_experience"] = _numeric_match(
        predicted.years_of_experience, expected.years_of_experience, years_tolerance
    )
    matches["target_salary_eur"] = _numeric_match(
        predicted.target_salary_eur, expected.target_salary_eur, salary_tolerance
    )

    predicted_skills = {skill.casefold() for skill in predicted.skills}
    expected_skills = {skill.casefold() for skill in expected.skills}
    exact = predicted_skills == expected_skills
    if not predicted_skills and not expected_skills:
        precision = recall = f1 = 1.0
    elif not predicted_skills or not expected_skills:
        precision = recall = f1 = 0.0
    else:
        true_positives = len(predicted_skills & expected_skills)
        precision = true_positives / len(predicted_skills)
        recall = true_positives / len(expected_skills)
        f1 = 2 * true_positives / (len(predicted_skills) + len(expected_skills))

    return ProfileEvaluation(
        field_matches=matches,
        skills_precision=precision,
        skills_recall=recall,
        skills_f1=f1,
        skills_exact_match=exact,
        profile_exact_match=all(matches.values()) and exact,
    )


def evaluate_dataset(
    pairs: Iterable[tuple[CandidateProfile, CandidateProfile]],
    *,
    years_tolerance: float = 0.0,
    salary_tolerance: int = 0,
) -> DatasetEvaluation:
    """Moyennes macro ; accuracy globale sur les six scalaires uniquement.

    Chaque couple est (prédit, attendu). Un ensemble vide est rejeté.
    Le profile exact match utilise les tolérances configurées pour les nombres.
    """
    _validate_tolerance(years_tolerance, "years_tolerance")
    _validate_tolerance(salary_tolerance, "salary_tolerance")
    scores = [
        evaluate_profile(
            predicted,
            expected,
            years_tolerance=years_tolerance,
            salary_tolerance=salary_tolerance,
        )
        for predicted, expected in pairs
    ]
    count = len(scores)
    if count == 0:
        raise ValueError("Au moins un couple de profils est nécessaire à l'évaluation.")

    field_accuracy = {
        field: sum(score.field_matches[field] for score in scores) / count
        for field in SCALAR_FIELDS
    }
    return DatasetEvaluation(
        example_count=count,
        field_accuracy=field_accuracy,
        overall_field_accuracy=sum(
            sum(score.field_matches.values()) for score in scores
        )
        / (count * len(SCALAR_FIELDS)),
        skills_macro_precision=sum(score.skills_precision for score in scores) / count,
        skills_macro_recall=sum(score.skills_recall for score in scores) / count,
        skills_macro_f1=sum(score.skills_f1 for score in scores) / count,
        skills_exact_match_accuracy=sum(score.skills_exact_match for score in scores)
        / count,
        profile_exact_match_accuracy=sum(score.profile_exact_match for score in scores)
        / count,
    )
