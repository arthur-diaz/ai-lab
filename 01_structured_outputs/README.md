# Structured outputs

Statut : **Work in progress**.

Objectif : extraire des informations structurées depuis du texte avec un LLM
et Pydantic. Le modèle `CandidateProfile`, disponible dans
`structured_outputs.models`, valide et normalise les informations d'un candidat.
L'intégration LLM viendra ultérieurement.

Stack prévue : Python 3.12, uv, Pydantic v2, pytest et Ruff.

## Installation

Avec uv installé, depuis la racine de `ai-lab` :

```powershell
cd 01_structured_outputs
uv sync
```

Le fichier `uv.lock` fixe les versions des dépendances. `.env.example` prépare
les variables de la future intégration LLM ; elles ne sont pas utilisées à ce stade.

## Développement

Depuis le dossier `01_structured_outputs` :

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Pour appliquer le formatage : `uv run ruff format .`.
