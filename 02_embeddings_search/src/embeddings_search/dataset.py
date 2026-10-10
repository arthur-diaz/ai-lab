"""Load and validate the synthetic retrieval dataset; no retrieval logic."""

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DatasetError(ValueError):
    """Invalid dataset content or unreadable dataset file."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    category: str


@dataclass(frozen=True)
class EvaluationQuery:
    id: str
    text: str
    relevance: dict[str, int]


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, object_pairs_hook=_unique_object)
                except ValueError as exc:
                    raise DatasetError(f"{path}:{number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise DatasetError(f"{path}:{number}: expected JSON object")
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc
    if not records:
        raise DatasetError(f"{path}: empty dataset")
    return records


def _text(record: dict, field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{context}: missing or invalid {field}")
    return value


def load_corpus(path: str | Path | None = None) -> list[Document]:
    """Load documents from the repository data directory or an injected path."""
    path = Path(path) if path is not None else DATA_DIR / "corpus/documents.jsonl"
    documents = []
    ids = set()
    for number, record in enumerate(_read_jsonl(path), 1):
        context = f"{path}:{number}"
        document_id = _text(record, "id", context)
        if document_id in ids:
            raise DatasetError(f"{context}: duplicate document id: {document_id}")
        ids.add(document_id)
        documents.append(
            Document(
                id=document_id,
                title=_text(record, "title", context),
                text=_text(record, "text", context),
                category=_text(record, "category", context),
            )
        )
    return documents


def load_evaluation_queries(
    queries_path: str | Path | None = None,
    relevance_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> list[EvaluationQuery]:
    """Join queries and judgments, checking references against the corpus.

    Grades are 0 (irrelevant), 1 (useful complement), and 2 (main answer).
    Omitted documents have grade 0. Every query requires a main answer.
    """
    queries_path = (
        Path(queries_path)
        if queries_path is not None
        else DATA_DIR / "evaluation/queries.jsonl"
    )
    relevance_path = (
        Path(relevance_path)
        if relevance_path is not None
        else DATA_DIR / "evaluation/relevance.jsonl"
    )
    document_ids = {document.id for document in load_corpus(corpus_path)}
    queries = {}
    for number, record in enumerate(_read_jsonl(queries_path), 1):
        context = f"{queries_path}:{number}"
        query_id = _text(record, "id", context)
        if query_id in queries:
            raise DatasetError(f"{context}: duplicate query id: {query_id}")
        queries[query_id] = _text(record, "text", context)

    judgments = {}
    for number, record in enumerate(_read_jsonl(relevance_path), 1):
        context = f"{relevance_path}:{number}"
        query_id = _text(record, "query_id", context)
        if query_id not in queries:
            raise DatasetError(f"{context}: unknown query id: {query_id}")
        if query_id in judgments:
            raise DatasetError(f"{context}: duplicate relevance query id: {query_id}")
        relevance = record.get("relevance")
        if not isinstance(relevance, dict):
            raise DatasetError(f"{context}: missing or invalid relevance mapping")
        for document_id, grade in relevance.items():
            if document_id not in document_ids:
                raise DatasetError(f"{context}: unknown document id: {document_id}")
            if type(grade) is not int or grade not in {0, 1, 2}:
                raise DatasetError(
                    f"{context}: invalid relevance grade for {document_id}: {grade!r}"
                )
        if 2 not in relevance.values():
            raise DatasetError(f"{context}: query {query_id} needs a grade 2 document")
        judgments[query_id] = relevance
    missing = queries.keys() - judgments.keys()
    if missing:
        raise DatasetError(
            f"Missing relevance for query ids: {', '.join(sorted(missing))}"
        )
    return [
        EvaluationQuery(query_id, text, judgments[query_id])
        for query_id, text in queries.items()
    ]
