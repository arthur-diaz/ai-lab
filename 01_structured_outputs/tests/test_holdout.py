import json

from structured_outputs.dataset import DATA_DIR, load_evaluation_dataset
from structured_outputs.models import CandidateProfile


def test_holdout_pairs_and_independence(tmp_path, monkeypatch):
    raw_path = DATA_DIR / "holdout" / "raw" / "candidates.jsonl"
    expected_path = DATA_DIR / "holdout" / "expected" / "candidates_expected.jsonl"
    development = load_evaluation_dataset()
    monkeypatch.chdir(tmp_path)

    examples = load_evaluation_dataset(raw_path, expected_path)
    raw = [
        json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    expected = [
        json.loads(line)
        for line in expected_path.read_text(encoding="utf-8").splitlines()
    ]

    ids = {item.id for item in examples}
    assert len(examples) == len(ids) == len(raw) == len(expected) == 25
    assert ids == {f"holdout_{index:03d}" for index in range(1, 26)}
    assert ids == {item["id"] for item in raw} == {item["id"] for item in expected}
    assert ids.isdisjoint(item.id for item in development)
    assert {item.expected.name for item in examples}.isdisjoint(
        item.expected.name for item in development
    )
    assert {item.text for item in examples}.isdisjoint(
        item.text for item in development
    )
    raw_by_id = {item["id"]: item["text"] for item in raw}
    expected_by_id = {
        item["id"]: CandidateProfile.model_validate(item["expected"])
        for item in expected
    }
    for example in examples:
        assert isinstance(example.expected, CandidateProfile)
        assert example.text == raw_by_id[example.id]
        assert example.expected == expected_by_id[example.id]


def test_holdout_category_coverage():
    examples = load_evaluation_dataset(
        DATA_DIR / "holdout" / "raw" / "candidates.jsonl",
        DATA_DIR / "holdout" / "expected" / "candidates_expected.jsonl",
    )
    profiles = [item.expected for item in examples]
    assert {item.employment_type for item in profiles} == {
        "CDI",
        "CDD",
        "Freelance",
        "Internship",
        "Apprenticeship",
        None,
    }
    assert any(item.years_of_experience == 0 for item in profiles)
    assert any(item.years_of_experience is None for item in profiles)
    assert any(item.years_of_experience == 2.5 for item in profiles)
    for field in ("job_title", "location", "target_salary_eur"):
        assert any(getattr(item, field) is None for item in profiles)
        assert any(getattr(item, field) is not None for item in profiles)
    assert any(not item.skills for item in profiles)
    by_id = {item.id: item.expected for item in examples}
    # Quelques annotations sensibles, sans figer les formulations du texte.
    assert by_id["holdout_004"].job_title == "analyste BI"
    assert by_id["holdout_008"].skills == ["Docker", "AWS"]
    assert by_id["holdout_009"].skills == ["R"]
    assert by_id["holdout_016"].years_of_experience is None
    assert by_id["holdout_018"].target_salary_eur == 49000
    assert by_id["holdout_019"].target_salary_eur == 46000
    assert by_id["holdout_021"].location == "Caen"
