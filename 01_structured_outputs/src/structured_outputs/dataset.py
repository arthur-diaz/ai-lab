"""Chargement du dataset synthétique, sans exécution du LLM."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from structured_outputs.models import CandidateProfile

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DatasetError(ValueError):
    """Fichier JSONL invalide ou exemples impossibles à associer."""


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    text: str
    expected: CandidateProfile


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                context = f"{path}, ligne {line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(f"{context} : JSON invalide.") from error
                if not isinstance(record, dict):
                    raise DatasetError(f"{context} : objet JSON attendu.")
                identifier = record.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise DatasetError(f"{context} : id manquant ou invalide.")
                if identifier in records:
                    raise DatasetError(f"{context} : id dupliqué {identifier!r}.")
                records[identifier] = record
    except (OSError, UnicodeError) as error:
        raise DatasetError(f"Impossible de lire {path} en UTF-8.") from error
    return records


def load_evaluation_dataset(
    raw_path: str | Path = DATA_DIR / "raw" / "candidates.jsonl",
    expected_path: str | Path = DATA_DIR / "expected" / "candidates_expected.jsonl",
) -> list[EvaluationExample]:
    """Associer les fichiers par id, en conservant l'ordre du fichier brut."""
    raw = _read_jsonl(Path(raw_path))
    expected = _read_jsonl(Path(expected_path))
    if raw.keys() != expected.keys():
        raise DatasetError(
            "Ids non appariés : "
            f"sans expected={sorted(raw.keys() - expected.keys())}, "
            f"sans texte={sorted(expected.keys() - raw.keys())}."
        )

    examples: list[EvaluationExample] = []
    for identifier, record in raw.items():
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(
                f"{raw_path} : texte manquant ou vide pour {identifier!r}."
            )
        profile_data = expected[identifier].get("expected")
        if not isinstance(profile_data, dict):
            raise DatasetError(
                f"{expected_path} : objet expected manquant ou invalide "
                f"pour {identifier!r}."
            )
        try:
            profile = CandidateProfile.model_validate(profile_data)
        except ValidationError as error:
            raise DatasetError(
                f"{expected_path} : expected incompatible avec CandidateProfile "
                f"pour {identifier!r}."
            ) from error
        examples.append(EvaluationExample(id=identifier, text=text, expected=profile))
    return examples
