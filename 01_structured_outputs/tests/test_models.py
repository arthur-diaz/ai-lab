import pytest
from pydantic import ValidationError

from structured_outputs.models import CandidateProfile


def test_complete_profile():
    values = {
        "name": "Marie Dupont",
        "job_title": "Data engineer",
        "years_of_experience": 4.5,
        "skills": ["Python", "SQL"],
        "location": "Lyon",
        "employment_type": "CDI",
        "target_salary_eur": 55000,
    }

    assert CandidateProfile(**values).model_dump() == values


def test_empty_profile():
    assert CandidateProfile().model_dump() == {
        "name": None,
        "job_title": None,
        "years_of_experience": None,
        "skills": [],
        "location": None,
        "employment_type": None,
        "target_salary_eur": None,
    }


def test_default_skills_are_independent():
    first = CandidateProfile()
    second = CandidateProfile()

    first.skills.append("Python")

    assert second.skills == []


@pytest.mark.parametrize("field", ["years_of_experience", "target_salary_eur"])
def test_negative_values_are_rejected(field):
    with pytest.raises(ValidationError) as error:
        CandidateProfile(**{field: -1})

    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "greater_than_equal"


@pytest.mark.parametrize("value", [0, 1000000])
def test_nonnegative_values_are_accepted_without_upper_bound(value):
    profile = CandidateProfile(years_of_experience=value, target_salary_eur=value)

    assert profile.years_of_experience == value
    assert profile.target_salary_eur == value


def test_text_is_trimmed_without_changing_case():
    profile = CandidateProfile(
        name="  Marie Dupont  ",
        job_title="  Data Engineer  ",
        location=" Lyon ",
        employment_type=" CDI ",
    )

    assert profile.name == "Marie Dupont"
    assert profile.job_title == "Data Engineer"
    assert profile.location == "Lyon"
    assert profile.employment_type == "CDI"


@pytest.mark.parametrize("field", ["name", "job_title", "location", "employment_type"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_text_becomes_none(field, value):
    assert getattr(CandidateProfile(**{field: value}), field) is None


def test_skills_are_trimmed_filtered_and_deduplicated_in_order():
    profile = CandidateProfile(
        skills=[" Python ", "SQL", "python", "", "  scikit-learn ", "  ", "sql"]
    )

    assert profile.skills == ["Python", "SQL", "scikit-learn"]


def test_skills_do_not_merge_distinct_technology_names():
    profile = CandidateProfile(skills=["postgres", "postgresql", "PostgreSQL"])

    assert profile.skills == ["postgres", "postgresql"]


def test_empty_skills_are_valid():
    assert CandidateProfile(skills=[]).skills == []


@pytest.mark.parametrize(
    "value", ["CDI", "CDD", "Freelance", "Internship", "Apprenticeship"]
)
def test_allowed_employment_types(value):
    assert CandidateProfile(employment_type=value).employment_type == value


@pytest.mark.parametrize("value", ["Permanent", "cdi"])
def test_other_employment_types_are_rejected(value):
    with pytest.raises(ValidationError) as error:
        CandidateProfile(employment_type=value)

    assert error.value.errors()[0]["loc"] == ("employment_type",)
    assert error.value.errors()[0]["type"] == "literal_error"
