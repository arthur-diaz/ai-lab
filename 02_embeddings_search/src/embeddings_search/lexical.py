"""Deterministic lexical retrieval using TF-IDF and cosine similarity."""

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from embeddings_search.dataset import Document


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    rank: int


class LexicalSearchEngine:
    """Index title and body once; preserve corpus order for equal scores."""

    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("Corpus must not be empty")
        self._documents = tuple(documents)
        self._vectorizer = TfidfVectorizer(lowercase=True)
        self._document_matrix = self._vectorizer.fit_transform(
            [f"{document.title}\n{document.text}" for document in self._documents]
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k documents, including zero-score matches.

        Empty queries and nonpositive or noninteger top_k values raise ValueError.
        A corpus with no token accepted by TF-IDF raises ValueError at indexing.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._document_matrix)[0]
        ordered = sorted(range(len(self._documents)), key=lambda index: -scores[index])
        return [
            SearchResult(self._documents[index], float(scores[index]), rank)
            for rank, index in enumerate(ordered[:top_k], start=1)
        ]
