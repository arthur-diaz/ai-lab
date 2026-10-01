import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from openai import APIConnectionError, BadRequestError, OpenAI, RateLimitError
from pydantic import ValidationError

from structured_outputs import CandidateExtractor, CandidateProfile, ExtractionError
from structured_outputs.config import AppConfig


@pytest.fixture
def client():
    client = Mock()
    client.with_options.return_value = client
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=CandidateProfile(name="Marie Dupont", skills=["Python"]),
        model="provider-model",
        usage=SimpleNamespace(input_tokens=120, output_tokens=35),
    )
    return client


def test_success_and_request(client, monkeypatch):
    timer = iter([10.0, 10.25])
    monkeypatch.setattr(
        "structured_outputs.extractor.perf_counter", lambda: next(timer)
    )
    extractor = CandidateExtractor(client, AppConfig(model="configured-model"))

    result = extractor.extract("Marie Dupont connaît Python.")

    assert result.profile == CandidateProfile(name="Marie Dupont", skills=["Python"])
    assert result.model == "provider-model"
    assert result.latency_seconds == 0.25
    assert (result.input_tokens, result.output_tokens) == (120, 35)
    assert result.estimated_cost_usd is None
    request = client.responses.parse.call_args.kwargs
    assert request["model"] == "configured-model"
    assert request["text_format"] is CandidateProfile
    assert request["input"][0]["role"] == "system"
    assert request["input"][1] == {
        "role": "user",
        "content": "Marie Dupont connaît Python.",
    }


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_does_not_call_provider(client, text):
    extractor = CandidateExtractor(client, AppConfig())

    with pytest.raises(ValueError, match="vide"):
        extractor.extract(text)

    client.responses.parse.assert_not_called()


@pytest.mark.parametrize(
    "metadata,input_tokens",
    [
        ({}, None),
        ({"usage": None}, None),
        ({"usage": SimpleNamespace(input_tokens=12)}, 12),
    ],
)
def test_missing_metadata(client, metadata, input_tokens):
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=CandidateProfile(), **metadata
    )

    result = CandidateExtractor(client, AppConfig()).extract("Profil inconnu.")

    assert result.model == "gpt-4o-mini"
    assert result.input_tokens == input_tokens
    assert result.output_tokens is None


@pytest.mark.parametrize(
    "error_type,status",
    [(BadRequestError, 400), (RateLimitError, 429)],
)
def test_provider_errors(client, error_type, status):
    response = httpx2.Response(
        status, request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )
    original = error_type("Provider error", response=response, body=None)
    client.responses.parse.side_effect = original

    with pytest.raises(ExtractionError, match=f"HTTP {status}") as error:
        CandidateExtractor(client, AppConfig()).extract("Profil candidat.")

    assert error.value.kind == "provider"
    assert error.value.__cause__ is original
    client.responses.parse.assert_called_once()  # Aucun retry local.


def test_connection_failure(client):
    client.responses.parse.side_effect = APIConnectionError(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )

    with pytest.raises(ExtractionError) as error:
        CandidateExtractor(client, AppConfig()).extract("Profil candidat.")

    assert error.value.kind == "provider"


def test_absent_structured_output(client):
    client.responses.parse.return_value = SimpleNamespace(output_parsed=None)

    with pytest.raises(ExtractionError, match="Profil structuré absent") as error:
        CandidateExtractor(client, AppConfig()).extract("Profil candidat.")

    assert error.value.kind == "structured_output"


def test_invalid_structured_output(client):
    with pytest.raises(ValidationError) as invalid:
        CandidateProfile(years_of_experience=-1)
    client.responses.parse.side_effect = invalid.value

    with pytest.raises(ExtractionError, match="structured output valide") as error:
        CandidateExtractor(client, AppConfig()).extract("Profil candidat.")

    assert error.value.kind == "structured_output"


def test_sdk_retry_and_timeout_configuration(client):
    CandidateExtractor(client, AppConfig(max_retries=1, timeout_seconds=12.5))

    client.with_options.assert_called_once_with(max_retries=1, timeout=12.5)


def test_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")

    config = AppConfig.from_env()

    assert config.api_key.get_secret_value() == "test-key"
    assert "test-key" not in repr(config)
    assert config.model == "custom-model"
    with config.create_client() as sdk_client:
        assert sdk_client.max_retries == 1
        assert sdk_client.timeout == 12.5


def test_configuration_defaults(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_MAX_RETRIES",
        "OPENAI_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = AppConfig.from_env()

    assert config.api_key is None
    assert config.model == "gpt-4o-mini"
    assert config.max_retries == 2
    assert config.timeout_seconds == 30


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_real_client_requires_key(api_key):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AppConfig(api_key=api_key).create_client()


@pytest.mark.parametrize(
    "values",
    [
        {"max_retries": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"model": "   "},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        AppConfig(**values)


def test_native_sdk_parsing_without_network():
    """Vérifier le schéma envoyé et le parsing réel avec un transport simulé."""

    def respond(request):
        body = json.loads(request.content)
        schema = body["text"]["format"]
        assert schema["type"] == "json_schema"
        assert schema["strict"] is True
        return httpx2.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        **CandidateProfile().model_dump(),
                                        "name": " Marie ",
                                        "skills": ["Python", "python"],
                                    }
                                ),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as http_client:
        with OpenAI(api_key="test-key", http_client=http_client) as sdk_client:
            result = CandidateExtractor(sdk_client, AppConfig()).extract(
                "Marie : Python"
            )

    assert isinstance(result.profile, CandidateProfile)
    assert result.profile.name == "Marie"
    assert result.profile.skills == ["Python"]
