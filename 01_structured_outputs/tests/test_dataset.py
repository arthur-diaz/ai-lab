import json

import pytest
from pydantic import ValidationError

from structured_outputs.dataset import DatasetError, load_evaluation_dataset
from structured_outputs.models import CandidateProfile


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def paths(tmp_path):
    raw = tmp_path / "raw.jsonl"
    expected = tmp_path / "expected.jsonl"
    write_jsonl(raw, [{"id": "example", "text": "Profil fictif."}])
    write_jsonl(expected, [{"id": "example", "expected": {"name": "Éloïse"}}])
    return raw, expected


def test_project_dataset_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    examples = load_evaluation_dataset()

    assert len(examples) == 15
    assert len({example.id for example in examples}) == 15
    assert [example.id for example in examples] == [
        f"candidate_{index:03d}" for index in range(1, 16)
    ]
    assert all(isinstance(example.expected, CandidateProfile) for example in examples)
    assert all(example.text.strip() for example in examples)


def test_dataset_annotations():
    profiles = {example.id: example.expected for example in load_evaluation_dataset()}

    assert profiles["candidate_002"].skills == []
    assert profiles["candidate_003"].target_salary_eur is None
    assert profiles["candidate_004"].location is None
    assert profiles["candidate_005"].years_of_experience == 2.5
    assert profiles["candidate_006"].employment_type == "Freelance"
    assert profiles["candidate_007"].employment_type == "CDD"
    assert profiles["candidate_008"].employment_type == "Internship"
    assert profiles["candidate_009"].employment_type == "Apprenticeship"
    assert profiles["candidate_011"].skills == ["Python", "SQL"]
    assert profiles["candidate_012"].skills == ["Python"]
    assert profiles["candidate_013"].target_salary_eur == 52000
    assert profiles["candidate_014"].location == "Angers"
    assert profiles["candidate_015"].years_of_experience is None
    assert profiles["candidate_015"].location is None
    assert profiles["candidate_015"].employment_type is None


def test_join_by_id_preserves_raw_order_and_unicode(paths):
    raw, expected = paths
    write_jsonl(
        raw,
        [{"id": "b", "text": "Deuxième."}, {"id": "a", "text": "Premier."}],
    )
    write_jsonl(
        expected,
        [
            {"id": "a", "expected": {"name": "A"}},
            {"id": "b", "expected": {"name": "Éloïse"}},
        ],
    )
    raw.write_text("\n" + raw.read_text(encoding="utf-8") + " \n", encoding="utf-8")

    examples = load_evaluation_dataset(raw, expected)

    assert [example.id for example in examples] == ["b", "a"]
    assert examples[0].expected.name == "Éloïse"
    assert examples[0].text == "Deuxième."


@pytest.mark.parametrize("side", [0, 1])
def test_invalid_json(paths, side):
    paths[side].write_text("\n{broken}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="ligne 2 : JSON invalide"):
        load_evaluation_dataset(*paths)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("record", [{}, {"id": None}, {"id": 1}, {"id": "   "}])
def test_missing_or_invalid_id(paths, side, record):
    write_jsonl(paths[side], [record])

    with pytest.raises(DatasetError, match="id manquant ou invalide"):
        load_evaluation_dataset(*paths)


@pytest.mark.parametrize("side", [0, 1])
def test_duplicate_id(paths, side):
    write_jsonl(paths[side], [{"id": "example"}, {"id": "example"}])

    with pytest.raises(DatasetError, match="id dupliqué 'example'"):
        load_evaluation_dataset(*paths)


@pytest.mark.parametrize("side", [0, 1])
def test_unmatched_ids(paths, side):
    write_jsonl(paths[side], [{"id": "other"}])

    with pytest.raises(DatasetError, match="Ids non appariés") as error:
        load_evaluation_dataset(*paths)

    assert "example" in str(error.value)
    assert "other" in str(error.value)


@pytest.mark.parametrize("profile", [{"years_of_experience": -1}, {"skills": [1]}])
def test_invalid_expected_profile(paths, profile):
    write_jsonl(paths[1], [{"id": "example", "expected": profile}])

    with pytest.raises(DatasetError, match="expected incompatible") as error:
        load_evaluation_dataset(*paths)

    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize("profile", [None, [], "profile"])
def test_missing_or_nonobject_expected(paths, profile):
    write_jsonl(paths[1], [{"id": "example", "expected": profile}])

    with pytest.raises(DatasetError, match="objet expected manquant ou invalide"):
        load_evaluation_dataset(*paths)


@pytest.mark.parametrize("text", [None, "", "   ", 12])
def test_missing_or_invalid_text(paths, text):
    write_jsonl(paths[0], [{"id": "example", "text": text}])

    with pytest.raises(DatasetError, match="texte manquant ou vide"):
        load_evaluation_dataset(*paths)


@pytest.mark.parametrize("side", [0, 1])
def test_nonobject_record(paths, side):
    write_jsonl(paths[side], [[]])

    with pytest.raises(DatasetError, match="objet JSON attendu"):
        load_evaluation_dataset(*paths)


def test_unreadable_file(paths, tmp_path):
    with pytest.raises(DatasetError, match="Impossible de lire"):
        load_evaluation_dataset(tmp_path / "missing.jsonl", paths[1])
