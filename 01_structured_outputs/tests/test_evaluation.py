import pytest

from structured_outputs import CandidateProfile, evaluate_dataset, evaluate_profile


@pytest.mark.parametrize("field", ["name", "job_title", "location"])
@pytest.mark.parametrize(
    "predicted,expected,strict,normalized",
    [
        ("Développeur Java", "Développeur Java", True, True),
        ("Développeur Java", "développeur Java", False, True),
        ("  Développeur Java  ", "développeur Java", False, True),
        ("technicien réseau", "technicien réseau en apprentissage", False, False),
        (None, None, True, True),
        (None, "Lyon", False, False),
        ("Lyon", None, False, False),
    ],
)
def test_normalized_text_metrics(field, predicted, expected, strict, normalized):
    score = evaluate_profile(
        CandidateProfile(**{field: predicted}), CandidateProfile(**{field: expected})
    )
    assert score.field_matches[field] is strict
    assert score.normalized_field_matches[field] is normalized
    assert score.profile_exact_match is strict
    assert score.normalized_profile_exact_match is normalized


@pytest.mark.parametrize("difference", ["skills", "salary", "employment"])
def test_normalized_profile_keeps_other_contracts(difference):
    expected = CandidateProfile(
        name="Marie", skills=["Python"], target_salary_eur=50000, employment_type="CDI"
    )
    changes = {
        "skills": {"skills": ["SQL"]},
        "salary": {"target_salary_eur": 50100},
        "employment": {"employment_type": "CDD"},
    }
    predicted = expected.model_copy(update={"name": "marie", **changes[difference]})
    score = evaluate_profile(predicted, expected)
    assert score.normalized_field_matches["name"]
    assert not score.normalized_profile_exact_match
    assert not score.profile_exact_match


def test_normalized_aggregation_and_numeric_tolerance():
    expected = CandidateProfile(name="Marie", years_of_experience=4)
    predicted = CandidateProfile(name="marie", years_of_experience=4.5)
    score = evaluate_dataset(
        [(predicted, expected), (expected, expected)], years_tolerance=0.5
    )
    assert score.overall_field_accuracy == pytest.approx(11 / 12)
    assert score.profile_exact_match_accuracy == 0.5
    assert score.normalized_overall_field_accuracy == 1
    assert score.normalized_profile_exact_match_accuracy == 1


@pytest.fixture
def complete_profile():
    return CandidateProfile(
        name="Marie",
        job_title="Développeuse",
        years_of_experience=4,
        location="Lyon",
        employment_type="CDI",
        target_salary_eur=55000,
        skills=["Python", "SQL"],
    )


def test_identical_profile(complete_profile):
    score = evaluate_profile(complete_profile, complete_profile.model_copy(deep=True))

    assert all(score.field_matches.values())
    assert score.skills_precision == score.skills_recall == score.skills_f1 == 1.0
    assert score.skills_exact_match
    assert score.profile_exact_match


def test_all_fields_differ(complete_profile):
    other = CandidateProfile(
        name="Jules",
        job_title="Designer",
        years_of_experience=1,
        location="Paris",
        employment_type="CDD",
        target_salary_eur=30000,
        skills=["Figma"],
    )

    score = evaluate_profile(other, complete_profile)

    assert not any(score.field_matches.values())
    assert score.skills_precision == score.skills_recall == score.skills_f1 == 0.0
    assert not score.skills_exact_match
    assert not score.profile_exact_match


def test_none_matches_none():
    score = evaluate_profile(CandidateProfile(), CandidateProfile())

    assert all(score.field_matches.values())
    assert score.profile_exact_match


@pytest.mark.parametrize("reverse", [False, True])
def test_none_against_value(complete_profile, reverse):
    pairs = (CandidateProfile(), complete_profile)
    if reverse:
        pairs = pairs[::-1]

    score = evaluate_profile(*pairs)

    assert not any(score.field_matches.values())
    assert not score.profile_exact_match


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "marie"),
        ("job_title", "développeuse"),
        ("location", "lyon"),
        ("job_title", "Développeur"),
    ],
)
def test_text_comparison_is_exact(complete_profile, field, value):
    predicted = complete_profile.model_copy(update={field: value})

    score = evaluate_profile(predicted, complete_profile)

    assert not score.field_matches[field]
    assert not score.profile_exact_match


@pytest.mark.parametrize(
    "predicted,expected,years_tolerance,salary_tolerance,match",
    [
        ((4, 55000), (4, 55000), 0, 0, True),
        ((4.25, 55100), (4, 55000), 0, 0, False),
        ((4.25, 55100), (4, 55000), 0.5, 200, True),
        ((4.5, 55200), (4, 55000), 0.5, 200, True),
        ((4.75, 55300), (4, 55000), 0.5, 200, False),
        ((None, None), (None, None), 10, 100000, True),
        ((None, None), (4, 55000), 10, 100000, False),
    ],
)
def test_numeric_comparison(
    predicted, expected, years_tolerance, salary_tolerance, match
):
    score = evaluate_profile(
        CandidateProfile(
            years_of_experience=predicted[0], target_salary_eur=predicted[1]
        ),
        CandidateProfile(
            years_of_experience=expected[0], target_salary_eur=expected[1]
        ),
        years_tolerance=years_tolerance,
        salary_tolerance=salary_tolerance,
    )

    assert score.field_matches["years_of_experience"] is match
    assert score.field_matches["target_salary_eur"] is match
    assert score.profile_exact_match is match


@pytest.mark.parametrize(
    "predicted,expected,precision,recall,f1,exact",
    [
        (["Python", "SQL"], ["Python", "SQL"], 1, 1, 1, True),
        (
            ["Python", "SQL", "Docker"],
            ["Python", "SQL", "AWS"],
            2 / 3,
            2 / 3,
            2 / 3,
            False,
        ),
        (["Python"], ["Python", "SQL", "AWS"], 1, 1 / 3, 0.5, False),
        (["Python", "SQL", "AWS"], ["Python"], 1 / 3, 1, 0.5, False),
        (["Docker"], ["Python"], 0, 0, 0, False),
        ([], [], 1, 1, 1, True),
        (["Python"], [], 0, 0, 0, False),
        ([], ["Python"], 0, 0, 0, False),
        (["python", "sql"], ["Python", "SQL"], 1, 1, 1, True),
        (["SQL", "Python"], ["Python", "SQL"], 1, 1, 1, True),
    ],
)
def test_skills_metrics(predicted, expected, precision, recall, f1, exact):
    score = evaluate_profile(
        CandidateProfile(skills=predicted), CandidateProfile(skills=expected)
    )

    assert score.skills_precision == pytest.approx(precision)
    assert score.skills_recall == pytest.approx(recall)
    assert score.skills_f1 == pytest.approx(f1)
    assert score.skills_exact_match is exact
    assert score.profile_exact_match is exact


def test_dataset_macro_aggregation():
    pairs = [
        (
            CandidateProfile(name="Marie", skills=["Python"]),
            CandidateProfile(name="Marie", skills=["Python"]),
        ),
        (
            CandidateProfile(name="Jules", skills=["Python", "SQL", "AWS"]),
            CandidateProfile(name="Marie", skills=["Python"]),
        ),
    ]

    score = evaluate_dataset(iter(pairs))

    assert score.example_count == 2
    assert score.field_accuracy == {
        "name": 0.5,
        "job_title": 1,
        "years_of_experience": 1,
        "location": 1,
        "employment_type": 1,
        "target_salary_eur": 1,
    }
    assert score.overall_field_accuracy == pytest.approx(11 / 12)
    assert score.skills_macro_precision == pytest.approx(2 / 3)
    assert score.skills_macro_recall == pytest.approx(1)
    assert score.skills_macro_f1 == pytest.approx(0.75)
    assert score.skills_exact_match_accuracy == pytest.approx(0.5)
    assert score.profile_exact_match_accuracy == pytest.approx(0.5)


def test_dataset_forwards_tolerances():
    predicted = CandidateProfile(years_of_experience=4.5, target_salary_eur=55100)
    expected = CandidateProfile(years_of_experience=4, target_salary_eur=55000)

    exact = evaluate_dataset([(predicted, expected)])
    tolerant = evaluate_dataset(
        [(predicted, expected)], years_tolerance=0.5, salary_tolerance=100
    )

    assert exact.overall_field_accuracy == pytest.approx(4 / 6)
    assert exact.profile_exact_match_accuracy == 0
    assert tolerant.overall_field_accuracy == 1
    assert tolerant.profile_exact_match_accuracy == 1


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="Au moins un couple"):
        evaluate_dataset([])


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("option", ["years_tolerance", "salary_tolerance"])
def test_invalid_tolerances(value, option):
    with pytest.raises(ValueError, match=option):
        evaluate_profile(CandidateProfile(), CandidateProfile(), **{option: value})
    with pytest.raises(ValueError, match=option):
        evaluate_dataset([], **{option: value})
