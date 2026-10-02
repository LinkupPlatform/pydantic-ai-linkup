"""Tests for `LinkupSearchToolset`: request parameters, output formatting and error mapping."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
from linkup import (
    LinkupAuthenticationError,
    LinkupFailedFetchError,
    LinkupInsufficientCreditError,
    LinkupNoResultError,
    LinkupTimeoutError,
    LinkupTooManyRequestsError,
)
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn

from pydantic_ai_linkup import LinkupSearch, LinkupSearchToolset
from tests.fakes import (
    FakeLinkupClient,
    fetch_response,
    image_result,
    search_results,
    sourced_answer,
    text_result,
)


def _text(output: ToolReturn[str]) -> str:
    """The model-facing text of a tool result."""
    body = output.return_value
    assert isinstance(body, str)
    return body


def _toolset(client: FakeLinkupClient, **kwargs: object) -> LinkupSearchToolset[None]:
    return LinkupSearch[None](client=client, **kwargs).get_toolset()  # type: ignore[arg-type]


class TestWebSearch:
    async def test_formats_results_with_content_and_sources(self) -> None:
        client = FakeLinkupClient(
            search_response=search_results(
                text_result("https://a.dev", name="A", content="alpha content"),
                text_result("https://b.dev", name="", content="beta content"),
            )
        )
        output = await _toolset(client).web_search("rust web frameworks")
        assert _text(output) == (
            "Found 2 results for 'rust web frameworks':\n\n"
            "Title: A\n"
            "URL: https://a.dev\n"
            "\n"
            "alpha content"
            "\n\n---\n\n"
            "Title: (untitled)\n"
            "URL: https://b.dev\n"
            "\n"
            "beta content"
        )
        assert output.metadata == {
            "sources": [
                {"url": "https://a.dev", "title": "A"},
                {"url": "https://b.dev", "title": None},
            ]
        }

    async def test_single_result_is_not_pluralized(self) -> None:
        client = FakeLinkupClient(search_response=search_results(text_result()))
        output = await _toolset(client).web_search("q")
        assert _text(output).startswith("Found 1 result for 'q':")

    async def test_forwards_configuration_to_linkup(self) -> None:
        client = FakeLinkupClient(search_response=search_results(text_result()))
        toolset = _toolset(
            client,
            depth="deep",
            max_results=3,
            include_domains=["a.dev"],
            exclude_domains=["b.dev"],
            from_date=date(2026, 1, 1),
            to_date=date(2026, 6, 30),
            timeout=30.0,
        )
        await toolset.web_search("q")
        assert client.search_calls == [
            {
                "query": "q",
                "depth": "deep",
                "output_type": "searchResults",
                "from_date": date(2026, 1, 1),
                "to_date": date(2026, 6, 30),
                "exclude_domains": ["b.dev"],
                "include_domains": ["a.dev"],
                "max_results": 3,
                "timeout": 30.0,
            }
        ]

    async def test_defaults_send_no_domain_or_date_filters(self) -> None:
        client = FakeLinkupClient(search_response=search_results(text_result()))
        await _toolset(client).web_search("q")
        call = client.search_calls[0]
        assert call["depth"] == "standard"
        assert call["max_results"] == 10
        assert call["timeout"] == 120.0
        assert call["include_domains"] is None
        assert call["exclude_domains"] is None
        assert call["from_date"] is None
        assert call["to_date"] is None

    async def test_caps_results_at_max_results_and_skips_images(self) -> None:
        client = FakeLinkupClient(
            search_response=search_results(
                image_result(),
                text_result("https://a.dev"),
                text_result("https://b.dev"),
                text_result("https://c.dev"),
            )
        )
        output = await _toolset(client, max_results=2).web_search("q")
        assert [source["url"] for source in output.metadata["sources"]] == [
            "https://a.dev",
            "https://b.dev",
        ]
        assert "image.png" not in _text(output)

    async def test_no_results(self) -> None:
        client = FakeLinkupClient(search_response=search_results(image_result()))
        output = await _toolset(client).web_search("nothing")
        assert _text(output) == "No results found for 'nothing'."
        assert output.metadata == {"sources": []}

    async def test_no_result_error_is_reported_as_empty_result(self) -> None:
        client = FakeLinkupClient(errors=[LinkupNoResultError("no result")])
        output = await _toolset(client).web_search("nothing")
        assert _text(output) == "No results found for 'nothing'."
        assert output.metadata == {"sources": []}


class TestGetPage:
    async def test_returns_markdown_with_source(self) -> None:
        client = FakeLinkupClient(fetch_response=fetch_response("# Title\n\nBody."))
        output = await _toolset(client).get_page("https://a.dev/post")
        assert _text(output) == "URL: https://a.dev/post\n\n# Title\n\nBody."
        assert output.metadata == {"sources": [{"url": "https://a.dev/post", "title": None}]}
        assert client.fetch_calls == [
            {"url": "https://a.dev/post", "render_js": False, "timeout": 120.0}
        ]

    async def test_forwards_render_js(self) -> None:
        client = FakeLinkupClient()
        await _toolset(client, render_js=True).get_page("https://a.dev")
        assert client.fetch_calls[0]["render_js"] is True

    async def test_truncates_long_pages_within_the_cap(self) -> None:
        client = FakeLinkupClient(fetch_response=fetch_response("x" * 500))
        output = await _toolset(client, max_text_chars=100).get_page("https://a.dev")
        body = _text(output).removeprefix("URL: https://a.dev\n\n")
        assert len(body) == 100
        assert body.endswith("[... page text truncated at 100 characters]")

    async def test_empty_page_asks_the_model_to_retry(self) -> None:
        client = FakeLinkupClient(fetch_response=fetch_response(""))
        with pytest.raises(ModelRetry, match="No content could be retrieved"):
            await _toolset(client).get_page("https://a.dev")


class TestAnswer:
    async def test_not_exposed_by_default(self) -> None:
        assert set(_toolset(FakeLinkupClient()).tools) == {"web_search", "get_page"}

    async def test_exposed_with_include_answer(self) -> None:
        toolset = _toolset(FakeLinkupClient(), include_answer=True)
        assert set(toolset.tools) == {"web_search", "get_page", "answer"}

    async def test_returns_answer_with_sources(self) -> None:
        client = FakeLinkupClient(
            answer_response=sourced_answer(
                "Linkup is a web search API.",
                ("https://www.linkup.so", "Linkup"),
                ("https://docs.linkup.so", ""),
            )
        )
        output = await _toolset(client, include_answer=True).answer("What is Linkup?")
        assert _text(output) == (
            "Linkup is a web search API.\n"
            "\n"
            "Sources:\n"
            "- Linkup: https://www.linkup.so\n"
            "- (untitled): https://docs.linkup.so"
        )
        assert output.metadata == {
            "sources": [
                {"url": "https://www.linkup.so", "title": "Linkup"},
                {"url": "https://docs.linkup.so", "title": None},
            ]
        }
        assert client.search_calls[0]["output_type"] == "sourcedAnswer"

    async def test_answer_without_sources(self) -> None:
        client = FakeLinkupClient(answer_response=sourced_answer("Just an answer."))
        output = await _toolset(client, include_answer=True).answer("q")
        assert _text(output) == "Just an answer."
        assert output.metadata == {"sources": []}

    async def test_empty_answer_asks_the_model_to_retry(self) -> None:
        client = FakeLinkupClient(answer_response=sourced_answer(""))
        with pytest.raises(ModelRetry, match="no answer"):
            await _toolset(client, include_answer=True).answer("q")


class TestErrors:
    @pytest.mark.parametrize(
        "error",
        [
            LinkupTooManyRequestsError("rate limited"),
            LinkupTimeoutError("timed out"),
            httpx.ConnectError("connection refused"),
        ],
    )
    async def test_transient_errors_become_retries(self, error: Exception) -> None:
        client = FakeLinkupClient(errors=[error])
        with pytest.raises(ModelRetry, match="Linkup request failed"):
            await _toolset(client).web_search("q")

    async def test_fetch_errors_become_retries(self) -> None:
        client = FakeLinkupClient(errors=[LinkupFailedFetchError("cannot fetch")])
        with pytest.raises(ModelRetry, match="cannot fetch"):
            await _toolset(client).get_page("https://a.dev")

    @pytest.mark.parametrize(
        "error",
        [LinkupAuthenticationError("bad key"), LinkupInsufficientCreditError("no credits")],
    )
    async def test_configuration_errors_propagate(self, error: Exception) -> None:
        client = FakeLinkupClient(errors=[error])
        with pytest.raises(type(error)):
            await _toolset(client).web_search("q")

    async def test_unexpected_errors_propagate(self) -> None:
        client = FakeLinkupClient(errors=[RuntimeError("bug")])
        with pytest.raises(RuntimeError, match="bug"):
            await _toolset(client).web_search("q")


class TestDefaultClient:
    def test_missing_api_key_raises_user_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LINKUP_API_KEY", raising=False)
        with pytest.raises(UserError, match="LINKUP_API_KEY"):
            LinkupSearch[None]().get_toolset()

    def test_builds_client_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINKUP_API_KEY", "test-key")
        toolset = LinkupSearch[None]().get_toolset()
        assert set(toolset.tools) == {"web_search", "get_page"}
