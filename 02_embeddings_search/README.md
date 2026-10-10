# Embeddings search

## Objectif

Explorer et évaluer la recherche lexicale et la recherche sémantique avec
embeddings sur un petit corpus documentaire. Le package et le dataset sont
prêts, avec une première baseline lexicale TF-IDF.

## Questions étudiées

- Quand les embeddings améliorent-ils la recherche par rapport au lexical ?
- Quels types de requêtes échouent avec chaque approche ?
- Comment mesurer objectivement la qualité du retrieval ?
- Quel compromis qualité / latence obtient-on ?

## Stack

Python 3.12, uv, Hatchling, scikit-learn, pytest et Ruff.
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

## Dataset

**Corpus** : 30 documents synthétiques en français pour Noria, un SaaS B2B
fictif. Chaque document de 50 à 150 mots constitue une unité de retrieval,
sans chunking. Les sujets voisins couvrent authentification, organisation,
facturation, API, webhooks, intégrations, données, sécurité, conformité et support.
Les politiques décrites sont fictives, pas des conseils sur un service réel.

**Evaluation queries** : 20 intentions distinctes, avec termes exacts,
paraphrases, vocabulaire utilisateur, formulations bruitées et besoins combinés.

**Graded relevance ground truth** : 2 signifie réponse principale, 1 complément
utile, 0 non pertinent. Les documents absents d'un jugement valent 0. Chaque
requête possède au moins un grade 2 ; un sujet voisin ne suffit pas à justifier
la pertinence. Par exemple, arrêter l'abonnement et emporter ses données demande
deux réponses principales ; récupérer un projet supprimé vise la Corbeille,
pas les sauvegardes globales. Les jugements précèdent tout résultat de moteur
et ne préjugent pas de l'approche qui réussira le mieux.

Les fichiers sont `data/corpus/documents.jsonl`, `data/evaluation/queries.jsonl`
et `data/evaluation/relevance.jsonl`. Depuis une installation editable dans ce
dépôt, les chemins par défaut sont indépendants du répertoire courant :

```python
from embeddings_search.dataset import load_corpus, load_evaluation_queries

documents = load_corpus()
queries = load_evaluation_queries()
```

`load_corpus(path=...)` et `load_evaluation_queries(queries_path=...,
relevance_path=..., corpus_path=...)` acceptent également des chemins explicites.
Les données restent dans le dépôt et ne sont pas embarquées dans le wheel.
Les loaders lèvent `DatasetError` pour les fichiers invalides, champs manquants,
doublons, références inconnues ou grades invalides.

## Baseline lexicale

`LexicalSearchEngine` construit une fois l'index TF-IDF sur le titre et le corps
de chaque document, sans pondération particulière. `TfidfVectorizer` utilise
ses réglages standard, dont la conversion en minuscules ; aucun stemming,
synonyme ou stopword personnalisé n'est ajouté.

Chaque requête est transformée avec le même vocabulaire, puis comparée aux
documents par similarité cosinus. Les résultats sont triés par score décroissant,
avec des rangs commençant à 1. Les égalités conservent l'ordre du corpus.

```python
from embeddings_search import LexicalSearchEngine
from embeddings_search.dataset import load_corpus

engine = LexicalSearchEngine(load_corpus())
for result in engine.search("comment changer mon mot de passe ?", top_k=5):
    print(result.rank, result.document.id, result.score)
```

Le corpus vide ou sans terme indexable est refusé à l'initialisation. Une requête
blanche et un `top_k` qui n'est pas un entier strictement positif lèvent
`ValueError`. Un `top_k` supérieur au corpus retourne tous les documents. Sans
terme commun, les scores sont nuls et l'ordre du corpus est conservé : un résultat
retourné n'est donc pas nécessairement pertinent.

Aucune métrique globale ni comparaison avec les embeddings n'est encore calculée.

## Validation

Depuis `02_embeddings_search`, sans appel API :

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import embeddings_search; print(embeddings_search.__name__)"
```

## Statut

Work in progress. Corpus, requêtes, annotations, loaders et baseline TF-IDF
disponibles. Aucun embedding, moteur sémantique, métrique ou benchmark implémenté.
