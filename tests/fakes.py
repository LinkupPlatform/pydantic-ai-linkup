"""Shared test doubles: an in-memory Linkup client returning real `linkup` SDK models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from linkup import (
    LinkupFetchResponse,
    LinkupSearchImageResult,
    LinkupSearchResults,
    LinkupSearchTextResult,
    LinkupSource,
    LinkupSourcedAnswer,
)


def text_result(
    url: str = "https://example.dev/page",
    *,
    name: str = "Example page",
    content: str = "Example content.",
) -> LinkupSearchTextResult:
    return LinkupSearchTextResult(type="text", name=name, url=url, content=content, favicon="")


def image_result(url: str = "https://example.dev/image.png") -> LinkupSearchImageResult:
    return LinkupSearchImageResult(type="image", name="An image", url=url)


def search_results(
    *results: LinkupSearchTextResult | LinkupSearchImageResult,
) -> LinkupSearchResults:
    return LinkupSearchResults(results=list(results))


def sourced_answer(answer: str, *sources: tuple[str, str]) -> LinkupSourcedAnswer:
    return LinkupSourcedAnswer(
        answer=answer,
        sources=[
            LinkupSource(name=name, url=url, snippet="snippet", favicon="") for url, name in sources
        ],
    )


def fetch_response(markdown: str = "# Example\n\nPage body.") -> LinkupFetchResponse:
    return LinkupFetchResponse(markdown=markdown, favicon="")


@dataclass
class FakeLinkupClient:
    """In-memory `LinkupClientProtocol` double: canned responses, recorded call arguments.

    `errors` are raised, in order, by the next calls before any canned response is returned.
    """

    search_response: LinkupSearchResults = field(default_factory=search_results)
    answer_response: LinkupSourcedAnswer = field(default_factory=lambda: sourced_answer(""))
    fetch_response: LinkupFetchResponse = field(default_factory=fetch_response)
    errors: list[Exception] = field(default_factory=list[Exception])
    search_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    fetch_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def async_search(self, query: str, **kwargs: Any) -> Any:
        self.search_calls.append({"query": query, **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        if kwargs["output_type"] == "sourcedAnswer":
            return self.answer_response
        return self.search_response

    async def async_fetch(self, url: str, **kwargs: Any) -> LinkupFetchResponse:
        self.fetch_calls.append({"url": url, **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        return self.fetch_response
