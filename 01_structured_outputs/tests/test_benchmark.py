import pytest

from structured_outputs import CandidateProfile, ExtractionError, ExtractionResult
from structured_outputs.benchmark import run_benchmark
from structured_outputs.dataset import EvaluationExample, load_evaluation_dataset
from structured_outputs.evaluation import evaluate_dataset, evaluate_profile


class FakeExtractor:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def extract(self, text):
        self.calls.append(text)
        result = next(self.results)
        if isinstance(result, ExtractionError):
            raise result
        return result


@pytest.fixture
def examples():
    return [
        EvaluationExample("second", "Profil Marie.", CandidateProfile(name="Marie")),
        EvaluationExample("first", "Profil Jules.", CandidateProfile(name="Jules")),
    ]


def test_perfect_benchmark_preserves_order_and_metadata(examples):
    extractor = FakeExtractor(
        [
            ExtractionResult(examples[0].expected, "fake-model", 0.25, 10, 3),
            ExtractionResult(examples[1].expected, "fake-model", 0.75, 20, 7),
        ]
    )

    result = run_benchmark(iter(examples), extractor)

    assert extractor.calls == [example.text for example in examples]
    assert [item.id for item in result.examples] == ["second", "first"]
    assert len(result.examples) == result.evaluation.example_count == 2
    assert result.evaluation.profile_exact_match_accuracy == 1
    assert all(value == 1 for value in result.evaluation.field_accuracy.values())
    assert result.evaluation.overall_field_accuracy == 1
    assert result.evaluation.skills_macro_precision == 1
    assert result.evaluation.skills_macro_recall == 1
    assert result.evaluation.skills_macro_f1 == 1
    assert result.evaluation.skills_exact_match_accuracy == 1
    assert result.total_input_tokens == 30
    assert result.total_output_tokens == 10
    assert result.total_latency_seconds == pytest.approx(1)
    assert result.mean_latency_seconds == pytest.approx(0.5)
    assert result.model == "fake-model"
    first = result.examples[0]
    assert first.expected == first.predicted == examples[0].expected
    assert first.evaluation.profile_exact_match
    assert first.latency_seconds == pytest.approx(0.25)
    assert (first.input_tokens, first.output_tokens, first.model) == (
        10,
        3,
        "fake-model",
    )


def test_imperfect_benchmark_reuses_evaluation(examples):
    predictions = [
        CandidateProfile(name="Other", skills=["Python"]),
        CandidateProfile(name="Jules", location="Paris"),
    ]
    extractor = FakeExtractor(
        [ExtractionResult(profile, "fake-model", 0.5) for profile in predictions]
    )

    result = run_benchmark(examples, extractor)

    pairs = list(zip(predictions, [example.expected for example in examples]))
    assert result.evaluation == evaluate_dataset(pairs)
    assert result.evaluation.overall_field_accuracy == pytest.approx(10 / 12)
    assert result.evaluation.profile_exact_match_accuracy == 0
    assert result.evaluation.skills_macro_f1 == pytest.approx(0.5)
    assert [item.evaluation for item in result.examples] == [
        evaluate_profile(predicted, expected) for predicted, expected in pairs
    ]


@pytest.mark.parametrize(
    "tokens,total_input,total_output",
    [
        ([(10, 3), (None, 7)], 10, 10),
        ([(None, 3), (None, None)], None, 3),
        ([(None, None), (None, None)], None, None),
        ([(0, 0), (None, None)], 0, 0),
    ],
)
def test_partial_or_missing_tokens(examples, tokens, total_input, total_output):
    extractor = FakeExtractor(
        [
            ExtractionResult(
                example.expected, "fake-model", 1, input_tokens, output_tokens
            )
            for example, (input_tokens, output_tokens) in zip(examples, tokens)
        ]
    )

    result = run_benchmark(examples, extractor)

    assert result.total_input_tokens == total_input
    assert result.total_output_tokens == total_output
    assert [
        (item.input_tokens, item.output_tokens) for item in result.examples
    ] == tokens


def test_mixed_models_are_explicit(examples):
    extractor = FakeExtractor(
        [
            ExtractionResult(examples[0].expected, "model-b", 1),
            ExtractionResult(examples[1].expected, "model-a", 1),
        ]
    )

    result = run_benchmark(examples, extractor)

    assert result.model == ("model-b", "model-a")
    assert [item.model for item in result.examples] == ["model-b", "model-a"]


def test_extraction_error_stops_benchmark(examples):
    failure = ExtractionError("Fake failure", kind="provider")
    extractor = FakeExtractor(
        [ExtractionResult(examples[0].expected, "fake-model", 1), failure]
    )

    with pytest.raises(ExtractionError) as error:
        run_benchmark([*examples, examples[0]], extractor)

    assert error.value is failure
    assert extractor.calls == [example.text for example in examples]


def test_empty_dataset_does_not_call_extractor():
    extractor = FakeExtractor([])

    with pytest.raises(ValueError, match="au moins un exemple"):
        run_benchmark([], extractor)

    assert extractor.calls == []


def test_real_dataset_with_fake_extractor():
    examples = load_evaluation_dataset()
    extractor = FakeExtractor(
        [ExtractionResult(example.expected, "fake-model", 1) for example in examples]
    )

    result = run_benchmark(examples, extractor)

    assert len(result.examples) == 15
    assert result.evaluation.profile_exact_match_accuracy == 1
    assert result.total_latency_seconds == pytest.approx(15)
    assert result.mean_latency_seconds == pytest.approx(1)
