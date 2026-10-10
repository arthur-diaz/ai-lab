# Embeddings search

## Objectif

Explorer et évaluer la recherche lexicale et la recherche sémantique avec
embeddings sur un petit corpus documentaire. Cette première étape prépare
uniquement le package Python et les outils de développement.

## Questions étudiées

- Quand les embeddings améliorent-ils la recherche par rapport au lexical ?
- Quels types de requêtes échouent avec chaque approche ?
- Comment mesurer objectivement la qualité du retrieval ?
- Quel compromis qualité / latence obtient-on ?

## Stack

Python 3.12, uv, Hatchling, pytest et Ruff. Aucune dépendance applicative.
Le package `embeddings_search` utilise un layout `src`.

## Installation

Avec uv installé, depuis la racine de `ai-lab` :

```bash
cd 02_embeddings_search
uv sync --locked
```

`.python-version` sélectionne Python 3.12 ; `uv.lock` fixe les dépendances.
Aucune clé API n'est nécessaire. `.env.example` prépare deux variables vides
pour un futur provider ; elles ne sont pas utilisées à cette étape.

## Validation

Depuis `02_embeddings_search`, sans appel API :

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import embeddings_search; print(embeddings_search.__name__)"
```

## Statut

Work in progress. Fondations uniquement : aucun corpus, moteur de recherche,
embedding ou système d'évaluation implémenté.
