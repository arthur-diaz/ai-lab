# Structured outputs

Statut : **Work in progress**.

Objectif : extraire des informations structurées depuis du texte avec un LLM
et Pydantic. Le modèle `CandidateProfile`, disponible dans
`structured_outputs.models`, valide et normalise les informations d'un candidat.
`CandidateExtractor` transforme un texte en profil avec le SDK OpenAI officiel.

Stack : Python 3.12, uv, Pydantic v2, SDK OpenAI, pytest et Ruff.

## Installation

Avec uv installé, depuis la racine de `ai-lab` :

```powershell
cd 01_structured_outputs
uv sync
```

Le fichier `uv.lock` fixe les versions des dépendances.

## Extraction en Python

Définir les variables dans l'environnement du processus (`.env.example` sert de
modèle ; aucun fichier `.env` n'est chargé automatiquement) :

- `OPENAI_API_KEY` : obligatoire pour créer un vrai client.
- `OPENAI_MODEL` : `gpt-4o-mini` par défaut.
- `OPENAI_MAX_RETRIES` : `2` par défaut, nombre de retries après l'appel initial.
- `OPENAI_TIMEOUT_SECONDS` : `30` par défaut, timeout du SDK par tentative,
  pas une limite sur la durée totale de l'extraction.

```python
from structured_outputs import CandidateExtractor, ExtractionError
from structured_outputs.config import AppConfig

config = AppConfig.from_env()
with config.create_client() as client:
    extractor = CandidateExtractor(client=client, config=config)
    try:
        result = extractor.extract("Marie Dupont, développeuse Python à Lyon.")
        print(result.profile.model_dump())
    except ExtractionError as error:
        print(error.kind, str(error))
```

L'extraction utilise `responses.parse` et `CandidateProfile` comme schéma
Pydantic, selon la [documentation officielle OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs).
Le prompt demande uniquement les informations explicites : les valeurs inconnues
restent `None`, et les compétences inconnues une liste vide.

Le client est injectable. Les retries sont exclusivement gérés par le SDK,
avec les paramètres de `AppConfig` appliqués au client injecté. Les erreurs
`ExtractionError` distinguent `provider` et `structured_output`, en conservant
l'exception d'origine comme cause. Le SDK décide seul des erreurs à rejouer.
Un texte vide provoque un `ValueError` local, sans appel API.

Le résultat contient le modèle retourné, la latence de l'appel (retries inclus),
les tokens si disponibles, et un coût estimé laissé à `None`.
Les tests utilisent des clients et transports simulés, sans réseau.
Aucune CLI n'est disponible à ce stade.

## Dataset synthétique

Les fichiers `data/raw/candidates.jsonl` et
`data/expected/candidates_expected.jsonl` contiennent 15 profils fictifs,
associés par des ids stables. Ils serviront à l'évaluation à l'étape suivante ;
aucune métrique ni exécution de benchmark n'est encore implémentée.

Les annotations distinguent compétences maîtrisées et simplement mentionnées,
salaire annuel cible et salaire actuel, résidence du candidat et adresse de
l'entreprise. Les informations absentes, inconnues ou indécidables restent
`null` (ou `[]` pour les compétences). Les intitulés reprennent le texte,
y compris le poste recherché lorsqu'il est explicitement indiqué.

```python
from structured_outputs.dataset import load_evaluation_dataset

examples = load_evaluation_dataset()
print(examples[0].id, examples[0].expected)
```

Le loader conserve l'ordre du fichier brut et ignore les lignes blanches.
Ses chemins par défaut ciblent les données du projet depuis le module, sans
dépendre du répertoire courant. Pour une installation sans le dossier `data/`,
fournir explicitement `raw_path` et `expected_path`.

## Développement

Depuis le dossier `01_structured_outputs` :

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Pour appliquer le formatage : `uv run ruff format .`.
