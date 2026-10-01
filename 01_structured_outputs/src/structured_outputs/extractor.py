"""Extraction synchrone via le parsing structuré natif du SDK OpenAI."""

from dataclasses import dataclass
from json import JSONDecodeError
from time import perf_counter
from typing import Literal

from openai import (
    APIError,
    APIResponseValidationError,
    APIStatusError,
    OpenAI,
)
from pydantic import ValidationError

from structured_outputs.config import AppConfig
from structured_outputs.models import CandidateProfile

SYSTEM_PROMPT = """Extrais uniquement les informations explicitement présentes dans le texte.
N'invente aucune valeur et n'infère aucune compétence absente.

Pour job_title, extrais uniquement l'intitulé du métier ou du poste,
sans y inclure le type de contrat ou les informations d'emploi qui disposent
déjà de leur propre champ.

Pour skills, retourne uniquement des noms concis de compétences, technologies,
outils ou domaines. Les mentions répétées d'une même compétence ne doivent
apparaître qu'une seule fois et les mots de liaison ou formulations autour
d'une compétence ne font pas partie de son nom.

Respecte le schéma : utilise null (None en Python) pour les informations absentes
et une liste vide pour les compétences inconnues."""


class ExtractionError(RuntimeError):
    """Échec d'extraction, catégorisé sans exposer le texte du provider."""

    def __init__(
        self,
        message: str,
        *,
        kind: Literal["provider", "structured_output"],
    ) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ExtractionResult:
    profile: CandidateProfile
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None


class CandidateExtractor:
    def __init__(self, client: OpenAI, config: AppConfig | None = None) -> None:
        self.config = config if config is not None else AppConfig.from_env()
        # Une seule stratégie : retries du SDK, jamais de boucle locale.
        self.client = client.with_options(
            max_retries=self.config.max_retries,
            timeout=self.config.timeout_seconds,
        )

    def extract(self, text: str) -> ExtractionResult:
        if not text.strip():
            raise ValueError("Le texte à extraire ne peut pas être vide.")

        # Le timer couvre l'appel complet, y compris les retries internes du SDK.
        start = perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.config.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                text_format=CandidateProfile,
            )
        except APIStatusError as error:
            raise ExtractionError(
                f"Échec du provider OpenAI (HTTP {error.status_code}).",
                kind="provider",
            ) from error
        except (
            ValidationError,
            JSONDecodeError,
            APIResponseValidationError,
        ) as error:
            raise ExtractionError(
                "La réponse ne contient pas de structured output valide.",
                kind="structured_output",
            ) from error
        except APIError as error:
            raise ExtractionError(
                "Échec du provider OpenAI.", kind="provider"
            ) from error
        latency = perf_counter() - start

        profile = response.output_parsed
        if not isinstance(profile, CandidateProfile):
            raise ExtractionError(
                "Profil structuré absent (refus ou réponse incomplète possible).",
                kind="structured_output",
            )
        usage = getattr(response, "usage", None)
        return ExtractionResult(
            profile=profile,
            model=getattr(response, "model", None) or self.config.model,
            latency_seconds=latency,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
