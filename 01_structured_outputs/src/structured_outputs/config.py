"""Configuration explicite à partir des variables d'environnement."""

import os

from openai import OpenAI
from pydantic import BaseModel, Field, SecretStr, field_validator


class AppConfig(BaseModel):
    api_key: SecretStr | None = None
    model: str = "gpt-4o-mini"
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le modèle OpenAI ne peut pas être vide.")
        return value

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Lire l'environnement sans charger automatiquement de fichier .env."""
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            max_retries=os.environ.get("OPENAI_MAX_RETRIES", "2"),
            timeout_seconds=os.environ.get("OPENAI_TIMEOUT_SECONDS", "30"),
        )

    def create_client(self) -> OpenAI:
        """Créer un vrai client ; l'appelant gère sa fermeture."""
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY est obligatoire pour un vrai appel.")
        return OpenAI(
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,
            timeout=self.timeout_seconds,
        )
