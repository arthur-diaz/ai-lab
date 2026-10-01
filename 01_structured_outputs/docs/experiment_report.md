# Structured Outputs — Experiment Report

## Objective

Extraire un `CandidateProfile` depuis un texte libre avec un LLM, puis mesurer
la justesse des valeurs, au-delà de leur conformité au schéma. Ce rapport
compare une baseline V1, une V2 améliorée sur le development set et le premier
passage de V2 sur un holdout indépendant de l'amélioration du prompt.

Les résultats ci-dessous proviennent des expériences communiquées et des
rapports JSON locaux. Aucun appel API n'a été relancé pour rédiger ce document.

## Experimental setup

- Modèle : `gpt-4o-mini`, identique pour les trois expériences.
- Python 3.12, uv, SDK OpenAI officiel, API Responses et parsing Pydantic natif.
- Schéma : nom, intitulé de poste, expérience, compétences, résidence, contrat
  et salaire annuel cible en euros.
- Pipeline : chargement JSONL → extraction séquentielle → scoring → rapport JSON.
- V2 précise la séparation de l'intitulé et du contrat, ainsi que les noms
  concis de compétences. Le schéma reste identique.
- Une exécution rapportée par expérience ; aucun benchmark répété ni comparaison
  de modèles. Les latences viennent de l'extractor et incluent ses retries SDK,
  sans inclure le scoring.

### Reproduction

Depuis `01_structured_outputs`, installer les versions verrouillées avec
`uv sync --locked`. Configurer l'environnement selon `.env.example` ; aucun
secret ne doit être committé. Les commandes suivantes sont des instructions
manuelles, effectuent de vrais appels et peuvent consommer des crédits API :

```powershell
# Development set, avec le prompt de la version sélectionnée
uv run --env-file .env python -m structured_outputs.cli benchmark --dataset development --run-api --output artifacts/development_reproduction.json

# Holdout, avec V2 gelée
uv run --env-file .env python -m structured_outputs.cli benchmark --dataset holdout --run-api --output artifacts/holdout_reproduction.json
```

Pour reproduire V1, utiliser son prompt et son état historique dans Git, plutôt
que le prompt V2 courant. Les scores normalisés peuvent aussi être recalculés
offline depuis les profils sauvegardés avec `rescore_report()` ; voir le README.
Cela ne recrée ni une extraction ni sa latence.

Les fichiers `baseline.json`, `baseline_normalized.json`, `v2.json` et
`holdout_v2.json` sous `artifacts/` restent locaux et ignorés par Git. Le rapport
publie les scores essentiels, pas les artifacts bruts. Les identifiants exacts
de commits, timestamps et paramètres effectifs de chaque run ne sont pas tous
consignés ici : les commandes reproduisent le protocole, sans garantir les mêmes
sorties. L'alias de modèle et la variabilité du service limitent également une
reproduction exacte.

## Development dataset vs holdout dataset

Le development set contient **15 exemples synthétiques**. L'analyse de ses
erreurs a guidé le prompt V2 : son score parfait normalisé est un résultat de
développement, **pas une estimation indépendante de généralisation**.

Le holdout contient **25 exemples synthétiques nouveaux**, avec candidats et
formulations distincts. Il n'a servi à aucun ajustement du prompt avant son
premier benchmark V2. Il fournit une estimation plus honnête, mais reste petit
et artificiel. Après analyse et adaptation sur ce jeu, un autre jeu indépendant
sera nécessaire pour une nouvelle estimation de généralisation.

## Metrics

- **Strict profile exact match** : les six scalaires correspondent et les
  ensembles de compétences sont identiques. Les textes sont sensibles à la casse.
- **Normalized profile exact match** : même règle, avec `strip().casefold()`
  uniquement sur `name`, `job_title` et `location`.
- **Scalar field accuracy** : comparaisons correctes / (6 × nombre d'exemples).
  Les skills sont exclues. La variante normalisée remplace uniquement les trois
  comparaisons textuelles ci-dessus.
- **Skills** : ensembles sans casse ni ordre. Précision = TP / prédictions,
  rappel = TP / attendus, F1 = 2 TP / (prédictions + attendus). Les scores macro
  sont calculés par candidat, puis moyennés. Le skills exact match exige des
  ensembles identiques.

Deux valeurs `None` correspondent ; une seule ne correspond pas. Les tolérances
numériques sont nulles dans ces benchmarks. Deux ensembles de compétences vides
reçoivent 1 pour précision, rappel et F1 ; un seul ensemble vide reçoit 0.

Le normalized match n'utilise ni fuzzy matching, ni synonymes, ni similarité
sémantique. Conserver les deux vues permet de distinguer une variation de
présentation d'une différence de contenu, sans effacer les erreurs métier.

## Baseline results

| Métrique | Development V1 | Development V2 | Holdout V2 |
|---|---:|---:|---:|
| Exemples | 15 | 15 | 25 |
| Strict profile exact match | 0.6000 (9/15) | 0.8667 (13/15) | 0.8800 (22/25) |
| Normalized profile exact match | 0.8667 (13/15) | 1.0000 (15/15) | 0.9200 (23/25) |
| Strict scalar field accuracy | 0.9444 (85/90) | 0.9778 (88/90) | 0.9800 (147/150) |
| Normalized scalar field accuracy | 0.9889 (89/90) | 1.0000 (90/90) | 0.9867 (148/150) |
| Skills macro precision | 0.9778 | 1.0000 | 1.0000 |
| Skills macro recall | 1.0000 | 1.0000 | 1.0000 |
| Skills macro F1 | 0.9867 | 1.0000 | 1.0000 |
| Skills exact-match accuracy | 0.9333 | 1.0000 | 1.0000 |
| Latence moyenne (s) | 2.402 | 2.768 | 2.187 |
| Latence totale (s) | 36.028 | 41.515 | 54.681 |
| Tokens d'entrée | 4756 | 6226 | 10272 |
| Tokens de sortie | 750 | 740 | 1220 |

Les métriques ont été vérifiées en lecture dans `baseline_normalized.json`,
`v2.json` et `holdout_v2.json`. Les valeurs décimales sont arrondies à quatre
chiffres et les latences à trois.

## V2 results

Le normalized profile exact match passe de **13/15 à 15/15**, soit **+13,33 points
de pourcentage**. Le skills macro F1 passe de **0.9867 à 1.0000**, soit environ
**+1,33 point**. Le strict profile exact match gagne **26,67 points** (9/15 → 13/15).
Les deux désaccords stricts restants concernent uniquement la casse.

## Holdout results

Sur les 25 nouveaux exemples, V2 obtient **92 % de normalized profile exact match**
et **100 % de skills macro F1**. Deux profils conservent un désaccord de fond ;
un troisième diffère seulement en casse. Le score scalar normalisé de 98,67 %
correspond à deux comparaisons incorrectes sur 150.

Ces observations ne permettent pas de qualifier le système de « 100 % précis ».
Le score de compétences parfait décrit ce petit holdout, pas toutes les entrées
possibles. Le development set et le holdout diffèrent aussi en contenu : leurs
scores ne doivent pas être comparés comme deux runs sur une population identique.

## Error analysis

### Development V1

Six profils échouent en strict : quatre différences de casse de `job_title`,
un intitulé contenant le type de contrat et une compétence supplémentaire
`encore Python`. La normalisation retire les quatre premiers désaccords, sans
cacher les deux erreurs de contenu. V2 corrige les deux erreurs de contenu.

### Holdout V2

| Id | Attendu → prédit | Classification | Interprétation |
|---|---|---|---|
| `holdout_004` | `analyste BI` → `Data Engineer` | Schema ambiguity | Le texte donne le poste actuel et le poste recherché ; la définition de `job_title` ne fixe pas leur priorité. |
| `holdout_006` | contrat `None` → `CDI` | Model error | Aucun contrat n'est présent : la prédiction introduit une information non supportée. |
| `holdout_022` | `développeur web` → `Développeur web` | Metric artifact | La différence de casse échoue en strict, mais correspond en normalized. |

Pour `holdout_004`, le ground truth retient le poste actuel selon la convention
documentée à la création du jeu. Le modèle peut raisonnablement privilégier le
poste recherché : ce désaccord révèle une faiblesse de spécification métier,
pas nécessairement une incapacité d'extraction. L'annotation et les scores sont
conservés, sans correction a posteriori ni exclusion de l'exemple.

## What improved

L'analyse a séparé les variations de présentation des erreurs actionnables.
La vue normalisée traite les premières ; le prompt V2 précise deux attentes
métier pour traiter les secondes. Le gain observé sur le development set est
cohérent avec ces corrections ciblées, sans prouver à lui seul leur généralisation.

## Trade-offs

Sur le même development set, la latence moyenne augmente de **0,366 s**
(2.402 → 2.768 s, **+15,2 %**), et les tokens d'entrée de **1470**
(4756 → 6226, **+30,9 %**). Les tokens de sortie baissent de 10 (750 → 740).
Le prompt enrichi s'accompagne donc d'un volume d'entrée plus élevé.

La variation de latence ne peut pas être attribuée uniquement au prompt : une
seule exécution ne contrôle ni la charge du service ni d'éventuels retries.
Les tokens et latences sont enregistrés pour une analyse de coût ultérieure.
Aucun tarif ni montant en dollars n'est calculé.

## Limitations

- Petits jeux synthétiques : vocabulaire et distributions éloignés de certains
  profils réels ; aucune validation en production.
- Development set utilisé pour ajuster le système ; son 100 % normalisé est
  optimiste comme estimation de généralisation.
- Un seul run par expérience, sans intervalle d'incertitude ni test de
  significativité : les améliorations et trade-offs restent descriptifs.
- Ambiguïté actuelle/recherchée de `job_title`, exposée par le holdout.
- Le match normalisé ignore seulement casse et espaces externes ; il peut
  encore pénaliser des formulations équivalentes, volontairement sans fuzzy matching.
- Les `None` correctement prédits et les skills vides contribuent aux scores
  selon les conventions annoncées ; les moyennes ne décrivent pas chaque cas.
- Traçabilité historique et modèle non figé par snapshot : reproductibilité
  du protocole plus forte que celle des résultats exacts.

## Lessons learned

1. Un résultat structuré et validé peut rester faux : `CDI` dans `holdout_006`
   respecte le type autorisé mais n'est pas justifié par le texte. Pydantic ne
   remplace pas une vérification sémantique.
2. La structure obtenue via Structured Outputs ne garantit pas la vérité des
   valeurs. Le scoring et l'analyse des exemples restent nécessaires.
3. Une mesure sensible à la casse peut sous-estimer le résultat utile : V1
   passe de 60 % strict à 86,67 % normalisé sans changer ses prédictions.
4. Analyser les erreurs avant d'ajuster le prompt évite de traiter la casse
   comme une erreur métier et permet des améliorations ciblées.
5. Réutiliser les erreurs d'un jeu pour développer le système le transforme
   en development set. Un holdout séparé est nécessaire pour estimer la généralisation.
6. Une spécification ambiguë peut limiter le système autant que le modèle,
   comme le montre la sélection du poste dans `holdout_004`.
7. Le gain de qualité doit être lu avec les ressources consommées : V2 améliore
   les scores de développement avec davantage de tokens d'entrée et une latence
   moyenne observée plus élevée.

## Next steps

Travaux possibles, non implémentés dans cette étape : clarifier la sémantique de
`job_title` (ou séparer poste actuel et poste recherché), préserver les résultats
historiques, constituer un nouveau test indépendant après tout ajustement, et
répéter les expériences avec une traçabilité plus complète des versions et runs.
La priorité est de préciser le contrat métier avant de chercher un score supérieur.
