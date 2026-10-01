"""CLI manuelle : aucun appel API sans --run-api."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from structured_outputs.benchmark import BenchmarkResult, run_benchmark
from structured_outputs.config import AppConfig
from structured_outputs.dataset import DATA_DIR, load_evaluation_dataset
from structured_outputs.evaluation import evaluate_dataset, evaluate_profile
from structured_outputs.extractor import CandidateExtractor, ExtractionError
from structured_outputs.models import CandidateProfile


def rescore_report(source: Path, output: Path) -> dict:
    """Recalculer offline les scores d'un rapport sans modifier sa source.

    Les latences, tokens et métadonnées historiques sont conservés. Les scores
    sont recalculés avec les tolérances par défaut, comme le benchmark runner.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Utilisez un chemin de sortie distinct du rapport historique.")
    report = json.loads(source.read_text(encoding="utf-8"))
    pairs = []
    for item in report["examples"]:
        predicted = CandidateProfile.model_validate(item["predicted"])
        expected = CandidateProfile.model_validate(item["expected"])
        pairs.append((predicted, expected))
        item.update(asdict(evaluate_profile(predicted, expected)))
    report["metrics"].update(asdict(evaluate_dataset(pairs)))
    report["metadata"]["rescored_at"] = datetime.now(UTC).isoformat()
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    return report


def _models(result: BenchmarkResult) -> list[str]:
    return [result.model] if isinstance(result.model, str) else list(result.model)


def _write_report(result: BenchmarkResult, path: Path) -> None:
    examples = []
    for item in result.examples:
        examples.append(
            {
                "id": item.id,
                "expected": item.expected.model_dump(mode="json"),
                "predicted": item.predicted.model_dump(mode="json"),
                **asdict(item.evaluation),
                "latency_seconds": item.latency_seconds,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "model": item.model,
            }
        )
    report = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "example_count": result.evaluation.example_count,
            "models": _models(result),
        },
        "metrics": {
            **asdict(result.evaluation),
            "total_latency_seconds": result.total_latency_seconds,
            "mean_latency_seconds": result.mean_latency_seconds,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        },
        "examples": examples,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _print_summary(result: BenchmarkResult) -> None:
    score = result.evaluation
    print("Benchmark completed\n")
    print(f"Examples: {score.example_count}")
    print(f"Models: {', '.join(_models(result))}\n")
    print(f"Strict profile exact match: {score.profile_exact_match_accuracy:.4f}")
    print(
        "Normalized profile exact match: "
        f"{score.normalized_profile_exact_match_accuracy:.4f}"
    )
    print(f"Strict scalar field accuracy: {score.overall_field_accuracy:.4f}")
    print(
        "Normalized scalar field accuracy: "
        f"{score.normalized_overall_field_accuracy:.4f}\n"
    )
    print(f"Skills macro precision: {score.skills_macro_precision:.4f}")
    print(f"Skills macro recall: {score.skills_macro_recall:.4f}")
    print(f"Skills macro F1: {score.skills_macro_f1:.4f}\n")
    print(f"Mean latency: {result.mean_latency_seconds:.3f} s")
    print(f"Total latency: {result.total_latency_seconds:.3f} s\n")
    print(f"Input tokens (known only): {result.total_input_tokens}")
    print(f"Output tokens (known only): {result.total_output_tokens}")
    mismatches = [
        item for item in result.examples if not item.evaluation.profile_exact_match
    ]
    if mismatches:
        print("\nMismatches:")
        for item in mismatches:
            fields = [
                name
                for name, match in item.evaluation.field_matches.items()
                if not match
            ]
            if not item.evaluation.skills_exact_match:
                fields.append("skills")
            print(f"- {item.id}: {', '.join(fields)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de profils candidats")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser(
        "benchmark", help="Lancer le benchmark manuellement"
    )
    benchmark.add_argument(
        "--run-api", action="store_true", help="Autoriser les appels OpenAI réels"
    )
    benchmark.add_argument("--output", type=Path, help="Chemin du rapport JSON")
    benchmark.add_argument(
        "--dataset",
        choices=["development", "holdout"],
        default="development",
        help="Jeu à évaluer (development par défaut)",
    )
    args = parser.parse_args(argv)
    if not args.run_api:
        print(
            "Dry run / mode protégé : aucun appel API. "
            "Utilisez --run-api pour lancer le benchmark réel."
        )
        print(f"Dataset: {args.dataset}")
        return 0
    try:
        config = AppConfig.from_env()
        # create_client valide la clé ; le contexte ferme le client même en échec.
        with config.create_client() as client:
            extractor = CandidateExtractor(client=client, config=config)
            if args.dataset == "holdout":
                examples = load_evaluation_dataset(
                    raw_path=DATA_DIR / "holdout" / "raw" / "candidates.jsonl",
                    expected_path=DATA_DIR
                    / "holdout"
                    / "expected"
                    / "candidates_expected.jsonl",
                )
            else:
                examples = load_evaluation_dataset()
            result = run_benchmark(examples=examples, extractor=extractor)
        if args.output is not None:
            _write_report(result, args.output)
    except ValidationError:
        # Ne pas afficher les valeurs d'environnement, notamment la clé API.
        print(
            "Erreur : configuration invalide. Vérifiez les variables OPENAI_*.",
            file=sys.stderr,
        )
        return 1
    except (ExtractionError, ValueError, OSError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
