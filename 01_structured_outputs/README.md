# Structured outputs

Mini-projet terminé : extraire des informations depuis du texte avec OpenAI
Structured Outputs, les valider avec Pydantic et évaluer leur justesse sur des
datasets synthétiques. Une structure valide ne garantit pas des valeurs exactes.

Stack : Python 3.12, uv, SDK OpenAI officiel, Pydantic v2, pytest et Ruff.
Le projet démontre le design de schéma, le parsing structuré natif, la validation,
les retries SDK, l'évaluation strict/normalized, l'analyse d'erreurs, les latences
et tokens, avec une suite de tests offline.

## Résultats

| Expérience | Exemples | Normalized profile exact match | Skills macro F1 |
|---|---:|---:|---:|
| Development V1 | 15 | 86,67 % | 98,67 % |
| Development V2 | 15 | 100 % | 100 % |
| Holdout V2 | 25 | 92 % | 100 % |

V2 a été optimisée sur le development set : son 100 % n'est pas une estimation
indépendante. Le premier passage sur le holdout est la mesure indépendante la
plus pertinente ici. Ces résultats ne signifient pas que le système est
« 100 % précis ». Voir le [rapport expérimental](docs/experiment_report.md)
pour le protocole, les scores stricts, les erreurs et les trade-offs.

## Installation

Avec uv installé, depuis la racine de `ai-lab` :

```powershell
cd 01_structured_outputs
uv sync --locked
```

`.python-version` sélectionne Python 3.12 et `uv.lock` fixe les dépendances.
Le package est installé depuis `src/`. Aucune clé API n'est nécessaire pour
l'installation, les tests ou les modes protégés.

## Configuration

Copier le modèle sans secret, puis renseigner la clé uniquement localement :

```powershell
Copy-Item .env.example .env
```

Ne pas écraser un `.env` existant. Variables disponibles :

| Variable | Valeur par défaut / rôle |
|---|---|
| `OPENAI_API_KEY` | Obligatoire pour construire un vrai client |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_MAX_RETRIES` | `2` retries SDK après l'appel initial |
| `OPENAI_TIMEOUT_SECONDS` | `30`, timeout par tentative, pas durée totale |

`AppConfig.from_env()` lit l'environnement du processus. Le package ne charge
pas automatiquement `.env` ; les commandes réelles ci-dessous le chargent avec
`uv --env-file`. `.env` et `artifacts/` restent locaux et ignorés par Git.

## CLI

Modes protégés, sans client ni appel API, même si une clé est présente :

```powershell
uv run python -m structured_outputs.cli benchmark
uv run python -m structured_outputs.cli benchmark --dataset holdout
```

Benchmarks réels, à déclencher manuellement uniquement. Ils effectuent des appels
OpenAI et peuvent consommer des crédits API :

```powershell
uv run --env-file .env python -m structured_outputs.cli benchmark --dataset development --run-api --output artifacts/development_reproduction.json
uv run --env-file .env python -m structured_outputs.cli benchmark --dataset holdout --run-api --output artifacts/holdout_reproduction.json
```

`development` est le dataset par défaut. `--output` est facultatif ; il crée les
répertoires parents et remplace le fichier choisi s'il existe. Utiliser un nouveau
nom pour préserver les rapports historiques. La console affiche un résumé et
uniquement les ids/champs en désaccord. Le JSON contient une date UTC, les modèles,
scores globaux, profils attendus/prédits et métadonnées individuelles.

La première erreur interrompt le benchmark avec un code non nul, sans rapport
complet ; aucun retry supplémentaire n'est ajouté par le runner. Un jeu vide
est rejeté. Les latences sont celles de l'extraction, retries SDK inclus. Les
tokens connus sont sommés ; le total vaut `None` si aucune valeur n'est connue
et peut être partiel sinon. Aucun prix en dollars n'est calculé.

## Datasets et métriques

- **Development** : 15 profils fictifs sous `data/raw/` et `data/expected/`,
  utilisés pour analyser les erreurs et améliorer le prompt.
- **Holdout** : 25 profils fictifs inédits sous `data/holdout/`, jamais utilisés
  pour ajuster le prompt avant leur premier benchmark. Après tout ajustement
  fondé sur ce jeu, il faudra un autre test indépendant.

Les fichiers JSONL sont associés par id. Les informations absentes restent
`None` (`[]` pour les compétences). Le loader détecte les ids invalides,
doublons, lignes invalides et profils incompatibles. Ses chemins par défaut sont
indépendants du répertoire courant ; une installation sans `data/` doit fournir
des chemins personnalisés.

`evaluate_profile(predicted, expected)` et `evaluate_dataset(pairs)` séparent :

- **Scalaires stricts** : valeurs validées comparées exactement, `None == None`
  correct ; tolérances numériques absolues inclusives, nulles par défaut.
- **Scalaires normalisés** : `strip().casefold()` seulement pour nom, poste et
  localisation, sans fuzzy matching ni synonymes. Les scores stricts restent disponibles.
- **Skills** : ensembles sans casse ni ordre, précision/rappel/F1 macro par
  candidat et exact match. Deux ensembles vides donnent 1 ; un seul vide donne 0.
- **Profile exact match** : les six scalaires et l'ensemble des skills doivent
  correspondre, selon la vue stricte ou normalisée. L'overall field accuracy
  porte uniquement sur les six scalaires.

Le recalcul d'un rapport historique est possible sans API et sans écraser la source :

```powershell
uv run python -c "from pathlib import Path; from structured_outputs.cli import rescore_report; rescore_report(Path('artifacts/baseline.json'), Path('artifacts/baseline_rescored.json'))"
```

Il réutilise les prédictions enregistrées, conserve les latences/tokens et ajoute
`rescored_at`, avec les tolérances par défaut.

## Architecture

| Module | Responsabilité |
|---|---|
| `models.py` | `CandidateProfile`, validation et normalisation métier |
| `config.py` | Variables OpenAI et construction explicite du client |
| `extractor.py` | `responses.parse`, erreurs applicatives et métadonnées |
| `dataset.py` | Chargement et association des JSONL |
| `evaluation.py` | Scoring individuel et agrégation |
| `benchmark.py` | Orchestration séquentielle avec extractor injectable |
| `cli.py` | Protection API, assemblage, console et rapports |

## Tests et qualité

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import structured_outputs; print(structured_outputs.__name__)"
```

Les 168 tests utilisent des fakes, mocks et un transport HTTP simulé : aucun
appel API réel. Pour appliquer le formatage : `uv run ruff format .`.

## Limites et Future improvements

Les jeux sont petits et synthétiques, avec une exécution par expérience. Les
scores et différences de latence sont descriptifs, sans preuve de significativité.
Le protocole est reproductible, mais les sorties peuvent varier et l'alias du
modèle n'est pas un snapshot figé. Pydantic ne détecte pas une inférence sémantique
non supportée, comme un contrat inventé.

La priorité entre poste actuel et recherché n'est pas fixée par `job_title` :
le ground truth de `holdout_004` retient le poste actuel. Cette convention et
son ambiguïté sont décrites dans le rapport, sans changer le holdout après coup.
Pour une version future : clarifier ce contrat métier, prévoir un nouveau test
indépendant et renforcer la traçabilité des runs. Le système V2 et les datasets
restent figés dans cette version.
