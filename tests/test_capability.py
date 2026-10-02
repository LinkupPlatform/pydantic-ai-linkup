"""Tests for the `LinkupSearch` capability: validation, instructions and agent specs."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec

from pydantic_ai_linkup import LinkupSearch
from tests.fakes import FakeLinkupClient


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"depth": "deepest"}, "depth must be one of"),
            ({"max_results": 0}, "max_results must be at least 1"),
            ({"max_text_chars": 0}, "max_text_chars must be at least 1"),
            (
                {"from_date": date(2026, 2, 1), "to_date": date(2026, 1, 1)},
                "must not be after to_date",
            ),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            LinkupSearch[None](**kwargs)  # type: ignore[arg-type]


class TestInstructions:
    def test_default_guidance(self) -> None:
        instructions = LinkupSearch[None](client=FakeLinkupClient()).get_instructions()
        assert instructions is not None
        assert "`web_search`" in instructions
        assert "`get_page`" in instructions
        assert "`answer`" not in instructions

    def test_guidance_mentions_answer_when_enabled(self) -> None:
        capability = LinkupSearch[None](client=FakeLinkupClient(), include_answer=True)
        instructions = capability.get_instructions()
        assert instructions is not None
        assert "`answer`" in instructions

    def test_custom_guidance_replaces_default(self) -> None:
        capability = LinkupSearch[None](client=FakeLinkupClient(), guidance="Only cite .gov sites.")
        assert capability.get_instructions() == "Only cite .gov sites."

    def test_empty_guidance_disables_instructions(self) -> None:
        assert LinkupSearch[None](client=FakeLinkupClient(), guidance="").get_instructions() is None


class TestAgentSpec:
    def test_spec_schema_includes_linkup_search(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([LinkupSearch])
        assert "LinkupSearch" in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        capability = LinkupSearch[None].from_spec(
            depth="deep",
            max_results=3,
            include_answer=True,
            include_domains=["a.dev"],
            from_date="2026-01-01",
            to_date=date(2026, 6, 30),
        )
        assert capability.depth == "deep"
        assert capability.max_results == 3
        assert capability.include_answer is True
        assert capability.include_domains == ["a.dev"]
        assert capability.from_date == date(2026, 1, 1)
        assert capability.to_date == date(2026, 6, 30)
        assert capability.client is None

    def test_agent_loads_from_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINKUP_API_KEY", "test-key")
        agent = Agent.from_spec(
            {"model": "test", "capabilities": [{"LinkupSearch": {"depth": "fast"}}]},
            custom_capability_types=[LinkupSearch],
        )
        assert isinstance(agent, Agent)
