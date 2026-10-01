"""Schéma métier des informations connues sur un candidat."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CandidateProfile(BaseModel):
    """Profil candidat dont les informations absentes restent explicites."""

    name: str | None = None
    job_title: str | None = None
    years_of_experience: float | None = Field(default=None, ge=0)
    skills: list[str] = Field(default_factory=list)
    location: str | None = None
    employment_type: (
        Literal["CDI", "CDD", "Freelance", "Internship", "Apprenticeship"] | None
    ) = None
    target_salary_eur: int | None = Field(default=None, ge=0)

    @field_validator("name", "job_title", "location", "employment_type", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        skills: list[str] = []
        seen: set[str] = set()
        for value in values:
            skill = value.strip()
            key = skill.casefold()
            if skill and key not in seen:
                skills.append(skill)
                seen.add(key)
        return skills
