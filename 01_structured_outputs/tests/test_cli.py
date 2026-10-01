import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from structured_outputs import CandidateProfile, ExtractionError, ExtractionResult, cli
from structured_outputs.benchmark import run_benchmark
from structured_outputs.dataset import DATA_DIR, EvaluationExample


@pytest.fixture
def assembled(monkeypatch):
    expected = CandidateProfile(name="Marie", skills=["Python"])
    examples = [EvaluationExample("candidate_001", "Marie : Python", expected)]
    extractor = Mock()
    extractor.extract.return_value = ExtractionResult(
        expected, "fake-model", 0.5, 10, 4
    )
    result = run_benchmark(examples, extractor)
    config = Mock()
    context = MagicMock()
    config.create_client.return_value = context
    client = context.__enter__.return_value
    from_env = Mock(return_value=config)
    constructor = Mock(return_value=extractor)
    loader = Mock(return_value=examples)
    runner = Mock(return_value=result)
    monkeypatch.setattr(cli.AppConfig, "from_env", from_env)
    monkeypatch.setattr(cli, "CandidateExtractor", constructor)
    monkeypatch.setattr(cli, "load_evaluation_dataset", loader)
    monkeypatch.setattr(cli, "run_benchmark", runner)
    return result, config, client, constructor, loader, runner


def test_protected_mode_with_key_never_assembles(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    from_env = Mock(side_effect=AssertionError("Configuration must not be read"))
    monkeypatch.setattr(cli.AppConfig, "from_env", from_env)
    forbidden = Mock(side_effect=AssertionError("No assembly in protected mode"))
    monkeypatch.setattr(cli, "CandidateExtractor", forbidden)
    monkeypatch.setattr(cli, "load_evaluation_dataset", forbidden)
    monkeypatch.setattr(cli, "run_benchmark", forbidden)
    output = tmp_path / "report.json"

    assert cli.main(["benchmark", "--output", str(output)]) == 0

    assert "mode protégé" in capsys.readouterr().out
    assert not output.exists()
    from_env.assert_not_called()
    forbidden.assert_not_called()


def test_fake_success_and_pipeline(assembled, capsys):
    result, config, client, constructor, loader, runner = assembled

    assert cli.main(["benchmark", "--run-api"]) == 0

    output = capsys.readouterr().out
    assert "Benchmark completed" in output
    assert "Examples: 1" in output
    assert "Models: fake-model" in output
    assert "Strict profile exact match: 1.0000" in output
    assert "Normalized profile exact match: 1.0000" in output
    assert "Strict scalar field accuracy: 1.0000" in output
    assert "Normalized scalar field accuracy: 1.0000" in output
    assert "Skills macro F1: 1.0000" in output
    assert "Mean latency: 0.500 s" in output
    assert "Input tokens (known only): 10" in output
    assert "Mismatches:" not in output
    constructor.assert_called_once_with(client=client, config=config)
    loader.assert_called_once_with()
    runner.assert_called_once_with(
        examples=loader.return_value, extractor=constructor.return_value
    )
    config.create_client.return_value.__exit__.assert_called_once()


def test_json_report(assembled, tmp_path):
    result = assembled[0]
    path = tmp_path / "nested" / "report.json"

    assert cli.main(["benchmark", "--run-api", "--output", str(path)]) == 0

    report = json.loads(path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(
        report["metadata"]["generated_at"]
    ).utcoffset() == timedelta(0)
    assert report["metadata"]["example_count"] == 1
    assert report["metadata"]["models"] == ["fake-model"]
    metrics = report["metrics"]
    assert metrics["overall_field_accuracy"] == 1
    assert metrics["normalized_overall_field_accuracy"] == 1
    assert metrics["normalized_profile_exact_match_accuracy"] == 1
    assert metrics["field_accuracy"]["name"] == 1
    assert metrics["skills_macro_precision"] == metrics["skills_macro_recall"] == 1
    assert metrics["skills_macro_f1"] == metrics["skills_exact_match_accuracy"] == 1
    assert metrics["profile_exact_match_accuracy"] == 1
    assert metrics["total_latency_seconds"] == metrics["mean_latency_seconds"] == 0.5
    assert (metrics["total_input_tokens"], metrics["total_output_tokens"]) == (10, 4)
    item = report["examples"][0]
    assert item["id"] == "candidate_001"
    assert (
        item["expected"]
        == item["predicted"]
        == result.examples[0].expected.model_dump()
    )
    assert item["field_matches"]["name"] is True
    assert item["normalized_field_matches"]["name"] is True
    assert item["normalized_profile_exact_match"] is True
    assert item["skills_exact_match"] is item["profile_exact_match"] is True
    assert item["skills_precision"] == item["skills_recall"] == item["skills_f1"] == 1
    assert item["latency_seconds"] == 0.5
    assert (item["input_tokens"], item["output_tokens"], item["model"]) == (
        10,
        4,
        "fake-model",
    )
    assert item["expected"]["location"] is None


def test_mismatches_only(assembled, capsys):
    expected = CandidateProfile(name="Marie", skills=["Python"])
    examples = [
        EvaluationExample("bad", "bad", expected),
        EvaluationExample("good", "good", expected),
    ]
    extractor = Mock()
    extractor.extract.side_effect = [
        ExtractionResult(CandidateProfile(name="Other"), "model-a", 1),
        ExtractionResult(expected, "model-b", 1),
    ]
    assembled[-1].return_value = run_benchmark(examples, extractor)

    assert cli.main(["benchmark", "--run-api"]) == 0

    output = capsys.readouterr().out
    assert "Models: model-a, model-b" in output
    assert "- bad: name, skills" in output
    assert "- good:" not in output
    assert "Other" not in output


def test_missing_key_does_not_launch(monkeypatch, capsys):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_MAX_RETRIES",
        "OPENAI_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = Mock(side_effect=AssertionError("No client without key"))
    monkeypatch.setattr("structured_outputs.config.OpenAI", provider)
    runner = Mock()
    monkeypatch.setattr(cli, "run_benchmark", runner)

    assert cli.main(["benchmark", "--run-api"]) == 1

    assert "OPENAI_API_KEY" in capsys.readouterr().err
    provider.assert_not_called()
    runner.assert_not_called()


def test_extraction_failure_no_success_or_report(assembled, tmp_path, capsys):
    assembled[-1].side_effect = ExtractionError(
        "Fake provider failure", kind="provider"
    )
    path = tmp_path / "report.json"

    assert cli.main(["benchmark", "--run-api", "--output", str(path)]) == 1

    captured = capsys.readouterr()
    assert "Benchmark completed" not in captured.out
    assert "Fake provider failure" in captured.err
    assert not path.exists()
    assembled[1].create_client.return_value.__exit__.assert_called_once()


def test_output_failure_returns_nonzero(assembled, tmp_path, capsys):
    assert cli.main(["benchmark", "--run-api", "--output", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Erreur" in captured.err
    assert "Benchmark completed" not in captured.out


def test_offline_rescore_preserves_history(assembled, tmp_path):
    source = tmp_path / "baseline.json"
    cli._write_report(assembled[0], source)
    report = json.loads(source.read_text(encoding="utf-8"))
    report["examples"][0]["predicted"]["name"] = "marie"
    source.write_text(json.dumps(report), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "rescored.json"

    result = cli.rescore_report(source, output)

    assert source.read_bytes() == original
    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["metadata"]["generated_at"] == report["metadata"]["generated_at"]
    assert result["metrics"]["total_input_tokens"] == 10
    assert result["metrics"]["profile_exact_match_accuracy"] == 0
    assert result["metrics"]["normalized_profile_exact_match_accuracy"] == 1
    assert not result["examples"][0]["field_matches"]["name"]
    assert result["examples"][0]["normalized_field_matches"]["name"]
    assembled[-1].assert_not_called()
    assembled[1].create_client.assert_not_called()


def test_offline_rescore_rejects_overwriting_source(tmp_path):
    source = tmp_path / "baseline.json"
    with pytest.raises(ValueError, match="distinct"):
        cli.rescore_report(source, source)


def test_explicit_development_dataset(assembled):
    assert cli.main(["benchmark", "--dataset", "development", "--run-api"]) == 0
    assembled[-2].assert_called_once_with()


def test_holdout_dataset_paths(assembled):
    assert cli.main(["benchmark", "--dataset", "holdout", "--run-api"]) == 0
    assembled[-2].assert_called_once_with(
        raw_path=DATA_DIR / "holdout" / "raw" / "candidates.jsonl",
        expected_path=DATA_DIR / "holdout" / "expected" / "candidates_expected.jsonl",
    )


def test_invalid_dataset_rejected(assembled):
    with pytest.raises(SystemExit) as error:
        cli.main(["benchmark", "--dataset", "invalid"])
    assert error.value.code == 2
    assembled[1].create_client.assert_not_called()
    assembled[-1].assert_not_called()


def test_holdout_dry_run_no_assembly(assembled, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    assert cli.main(["benchmark", "--dataset", "holdout"]) == 0
    output = capsys.readouterr().out
    assert "mode protégé" in output
    assert "Dataset: holdout" in output
    assembled[1].create_client.assert_not_called()
    assembled[-2].assert_not_called()
    assembled[-1].assert_not_called()
