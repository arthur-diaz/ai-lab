import pytest

from embeddings_search import LexicalSearchEngine, SearchResult
from embeddings_search.dataset import Document, load_corpus


@pytest.fixture
def documents():
    return [
        Document("mixed", "", "password invoice", "test"),
        Document("unrelated", "", "invoice billing", "test"),
        Document("exact", "", "password", "test"),
    ]


def test_obvious_match_ranking_ranks_and_scores(documents):
    results = LexicalSearchEngine(documents).search("password")
    assert [result.document.id for result in results] == ["exact", "mixed", "unrelated"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score > results[1].score > results[2].score
    assert all(0 <= result.score <= 1 + 1e-12 for result in results)
    assert all(isinstance(result, SearchResult) for result in results)
    assert all(isinstance(result.score, float) for result in results)


@pytest.mark.parametrize(("top_k", "count"), [(1, 1), (2, 2), (10, 3)])
def test_top_k(documents, top_k, count):
    results = LexicalSearchEngine(documents).search("password", top_k=top_k)
    assert len(results) == count
    assert [result.rank for result in results] == list(range(1, count + 1))


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_empty_query(documents, query):
    with pytest.raises(ValueError, match="Query"):
        LexicalSearchEngine(documents).search(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(documents, top_k):
    with pytest.raises(ValueError, match="top_k"):
        LexicalSearchEngine(documents).search("password", top_k)


def test_empty_corpus():
    with pytest.raises(ValueError, match="Corpus"):
        LexicalSearchEngine([])


def test_no_indexable_tokens():
    with pytest.raises(ValueError, match="empty vocabulary"):
        LexicalSearchEngine([Document("empty", "!", "?", "test")])


def test_unknown_terms_return_zero_scores_in_corpus_order(documents):
    engine = LexicalSearchEngine(documents)
    first = engine.search("xyzunknown")
    assert [result.document for result in first] == documents
    assert all(result.score == 0 for result in first)
    assert engine.search("xyzunknown") == first


def test_equal_positive_scores_preserve_corpus_order():
    documents = [Document(name, "", "password", "test") for name in ["z", "a", "m"]]
    results = LexicalSearchEngine(documents).search("password")
    assert [result.document.id for result in results] == ["z", "a", "m"]
    assert all(result.score > 0 for result in results)


def test_case_insensitive(documents):
    engine = LexicalSearchEngine(documents)
    assert engine.search("PASSWORD") == engine.search("password")


def test_title_and_body_are_indexed():
    documents = [
        Document("title", "authentication", "account", "test"),
        Document("body", "billing", "invoice", "test"),
    ]
    engine = LexicalSearchEngine(documents)
    assert engine.search("authentication", 1)[0].document.id == "title"
    assert engine.search("invoice", 1)[0].document.id == "body"


def test_index_is_reused_and_corpus_list_is_snapshotted(documents, monkeypatch):
    engine = LexicalSearchEngine(documents)
    documents.clear()

    def forbid_refit(*args, **kwargs):
        pytest.fail("The document index must not be rebuilt during search")

    monkeypatch.setattr(engine._vectorizer, "fit_transform", forbid_refit)
    assert engine.search("password", 1)[0].document.id == "exact"
    assert engine.search("billing", 1)[0].document.id == "unrelated"


def test_real_corpus_password_search():
    results = LexicalSearchEngine(load_corpus()).search("mot de passe", top_k=3)
    assert "doc_001" in {result.document.id for result in results}
