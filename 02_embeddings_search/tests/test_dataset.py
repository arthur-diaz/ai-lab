import json

import pytest

from embeddings_search.dataset import (
    DatasetError,
    load_corpus,
    load_evaluation_queries,
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def dataset_paths(tmp_path):
    corpus = write_jsonl(
        tmp_path / "documents.jsonl",
        [{"id": "doc", "title": "Title", "text": "Content", "category": "support"}],
    )
    queries = write_jsonl(
        tmp_path / "queries.jsonl", [{"id": "query", "text": "Question"}]
    )
    relevance = write_jsonl(
        tmp_path / "relevance.jsonl",
        [{"query_id": "query", "relevance": {"doc": 2}}],
    )
    return corpus, queries, relevance


def load_paths(paths):
    corpus, queries, relevance = paths
    return load_evaluation_queries(queries, relevance, corpus)


def test_real_corpus():
    documents = load_corpus()
    assert len(documents) == 30
    assert len({document.id for document in documents}) == 30
    assert all(
        document.title.strip() and document.text.strip() for document in documents
    )
    assert all(50 <= len(document.text.split()) <= 150 for document in documents)


def test_real_queries_and_ground_truth():
    queries = load_evaluation_queries()
    document_ids = {document.id for document in load_corpus()}
    assert len(queries) == 20
    assert len({query.id for query in queries}) == 20
    assert all(query.text.strip() for query in queries)
    for query in queries:
        assert 2 in query.relevance.values()
        assert query.relevance.keys() <= document_ids
        assert set(query.relevance.values()) <= {0, 1, 2}
    assert any(list(query.relevance.values()).count(2) > 1 for query in queries)
    assert any(1 in query.relevance.values() for query in queries)


def test_default_paths_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_corpus()) == 30
    assert len(load_evaluation_queries()) == 20


def test_custom_paths_join_query_and_relevance(dataset_paths):
    query = load_paths(dataset_paths)[0]
    assert (query.id, query.text, query.relevance) == ("query", "Question", {"doc": 2})


@pytest.mark.parametrize("index", [0, 1, 2])
def test_invalid_json(dataset_paths, index):
    dataset_paths[index].write_text("{broken\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_paths(dataset_paths)


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (0, "duplicate document id"),
        (1, "duplicate query id"),
        (2, "duplicate relevance query id"),
    ],
)
def test_duplicate_ids(dataset_paths, index, message):
    path = dataset_paths[index]
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_paths(dataset_paths)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"query_id": "unknown", "relevance": {"doc": 2}}, "unknown query id"),
        ({"query_id": "query", "relevance": {"unknown": 2}}, "unknown document id"),
        ({"query_id": "query", "relevance": {"doc": 1}}, "needs a grade 2"),
        ({"query_id": "query"}, "invalid relevance mapping"),
    ],
)
def test_invalid_judgments(dataset_paths, record, message):
    write_jsonl(dataset_paths[2], [record])
    with pytest.raises(DatasetError, match=message):
        load_paths(dataset_paths)


@pytest.mark.parametrize("grade", [-1, 3, 2.0, "2", True, None])
def test_invalid_grade(dataset_paths, grade):
    write_jsonl(dataset_paths[2], [{"query_id": "query", "relevance": {"doc": grade}}])
    with pytest.raises(DatasetError, match="invalid relevance grade"):
        load_paths(dataset_paths)


def test_missing_judgment(dataset_paths):
    write_jsonl(
        dataset_paths[1],
        [{"id": "query", "text": "Question"}, {"id": "other", "text": "Other"}],
    )
    with pytest.raises(DatasetError, match="Missing relevance.*other"):
        load_paths(dataset_paths)


@pytest.mark.parametrize("field", ["id", "title", "text", "category"])
@pytest.mark.parametrize("value", [None, "", "   ", 42])
def test_invalid_document_fields(dataset_paths, field, value):
    record = {"id": "doc", "title": "Title", "text": "Content", "category": "support"}
    record[field] = value
    write_jsonl(dataset_paths[0], [record])
    with pytest.raises(DatasetError, match=f"invalid {field}"):
        load_paths(dataset_paths)


@pytest.mark.parametrize("field", ["id", "text"])
def test_missing_query_fields(dataset_paths, field):
    record = {"id": "query", "text": "Question"}
    del record[field]
    write_jsonl(dataset_paths[1], [record])
    with pytest.raises(DatasetError, match=f"invalid {field}"):
        load_paths(dataset_paths)


def test_duplicate_json_relevance_key(dataset_paths):
    dataset_paths[2].write_text(
        '{"query_id":"query","relevance":{"doc":2,"doc":1}}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="duplicate JSON key: doc"):
        load_paths(dataset_paths)


def test_explicit_zero_grade(dataset_paths):
    documents = [
        {"id": name, "title": "Title", "text": "Content", "category": "support"}
        for name in ["doc", "other"]
    ]
    write_jsonl(dataset_paths[0], documents)
    write_jsonl(
        dataset_paths[2], [{"query_id": "query", "relevance": {"doc": 2, "other": 0}}]
    )
    assert load_paths(dataset_paths)[0].relevance == {"doc": 2, "other": 0}


@pytest.mark.parametrize("contents", ["", "\n", "[]\n"])
def test_empty_or_non_object_file(dataset_paths, contents):
    dataset_paths[0].write_text(contents, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_paths(dataset_paths)


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="Cannot read"):
        load_corpus(tmp_path / "missing.jsonl")
