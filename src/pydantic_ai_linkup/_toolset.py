"""Linkup toolset: web search, sourced answers and page fetching backed by the Linkup API."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date
from typing import Concatenate, Literal, ParamSpec, Protocol, TypeVar, overload

import httpx
import linkup
from linkup import (
    LinkupAuthenticationError,
    LinkupBudgetLimitExceededError,
    LinkupClient,
    LinkupFailedFetchError,
    LinkupFetchResponse,
    LinkupFetchResponseTooLargeError,
    LinkupFetchTargetNotFoundError,
    LinkupFetchTargetUnreachableError,
    LinkupFetchUnsupportedContentTypeError,
    LinkupInsufficientCreditError,
    LinkupInvalidRequestError,
    LinkupIpNotWhitelistedError,
    LinkupNoResultError,
    LinkupPaymentRequiredError,
    LinkupSearchResults,
    LinkupSearchTextResult,
    LinkupSourcedAnswer,
    LinkupTimeoutError,
    LinkupTooManyRequestsError,
    LinkupUnknownError,
)
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

SearchDepth = Literal["flash", "fast", "standard", "deep"]
"""Linkup search depth, from lowest latency (`flash`) to most thorough (`deep`)."""

DEFAULT_TIMEOUT_SECONDS = 120.0
"""Default per-request timeout in seconds; `deep` searches can take tens of seconds."""

_P = ParamSpec("_P")
_R = TypeVar("_R")
_SelfT = TypeVar("_SelfT")

# Authentication, authorization and billing states the model cannot correct: they propagate and
# abort the run instead of being fed back to the model as a retry prompt.
_PROPAGATE_ERRORS: tuple[type[Exception], ...] = (
    LinkupAuthenticationError,
    LinkupIpNotWhitelistedError,
    LinkupPaymentRequiredError,
    LinkupInsufficientCreditError,
    LinkupBudgetLimitExceededError,
)

# Failures a model can recover from by rephrasing, trying another URL, or simply retrying.
_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    LinkupInvalidRequestError,
    LinkupNoResultError,
    LinkupTooManyRequestsError,
    LinkupFailedFetchError,
    LinkupFetchResponseTooLargeError,
    LinkupFetchTargetNotFoundError,
    LinkupFetchTargetUnreachableError,
    LinkupFetchUnsupportedContentTypeError,
    LinkupTimeoutError,
    LinkupUnknownError,
    httpx.HTTPError,
    # Added in linkup-sdk 0.23.1; older versions raise `LinkupUnknownError` for the same 504.
    *(
        (linkup.LinkupRequestDeadlineExceededError,)
        if hasattr(linkup, "LinkupRequestDeadlineExceededError")
        else ()
    ),
)


class LinkupSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def _source_list(sources: Mapping[str, str | None]) -> list[LinkupSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{"url": url, "title": title} for url, title in sources.items()]


class LinkupClientProtocol(Protocol):
    """The subset of the `linkup.LinkupClient` API that `LinkupSearchToolset` calls.

    Any object with these methods can back the toolset. Pass one via `LinkupSearch.client` to
    configure the API key or base URL explicitly, or to substitute a fake in tests. The
    signatures mirror `LinkupClient`'s own, so a real `LinkupClient` satisfies the protocol as-is.
    """

    @overload
    async def async_search(
        self,
        query: str,
        *,
        depth: SearchDepth,
        output_type: Literal["searchResults"],
        from_date: date | None = None,
        to_date: date | None = None,
        exclude_domains: list[str] | None = None,
        include_domains: list[str] | None = None,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> LinkupSearchResults: ...

    @overload
    async def async_search(
        self,
        query: str,
        *,
        depth: SearchDepth,
        output_type: Literal["sourcedAnswer"],
        from_date: date | None = None,
        to_date: date | None = None,
        exclude_domains: list[str] | None = None,
        include_domains: list[str] | None = None,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> LinkupSourcedAnswer: ...

    async def async_fetch(
        self,
        url: str,
        *,
        render_js: bool | None = None,
        timeout: float | None = None,
    ) -> LinkupFetchResponse:
        """Fetch a web page and return its content as markdown."""
        ...  # pragma: no cover


def _default_client() -> LinkupClientProtocol:
    """Build a `LinkupClient` from the `LINKUP_API_KEY` environment variable."""
    try:
        return LinkupClient()
    except ValueError as error:
        raise UserError(
            "LinkupSearch needs a Linkup API key: set the LINKUP_API_KEY environment variable, "
            "or pass a configured client, e.g. LinkupSearch(client=LinkupClient(api_key=...)). "
            "Get a key at https://app.linkup.so."
        ) from error


def _recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient Linkup API failures into `ModelRetry`.

    Pydantic AI only feeds `ModelRetry` back to the model as a retry prompt; any other exception
    propagates and aborts the run. Rate limits, timeouts, rejected parameters and unreachable
    URLs are things a model can recover from, so they become retries. Authentication,
    authorization and credit errors are configuration states and propagate.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except _PROPAGATE_ERRORS:
            raise
        except _RECOVERABLE_ERRORS as error:
            raise ModelRetry(f"Linkup request failed: {error}") from error

    return wrapper


class LinkupSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Linkup API.

    `web_search` surveys the web and returns results with their content, and `get_page` fetches
    one specific URL as markdown. With `include_answer=True`, `answer` returns a synthesized
    answer to a question together with the sources supporting it.

    Each tool returns a `ToolReturn` whose `return_value` is the text the model sees and whose
    `metadata['sources']` lists the result URLs and titles (`LinkupSource` dicts), so
    applications can render citations from `ToolReturnPart.metadata` without parsing the text.

    `get_page` markdown is capped at `max_text_chars` characters. Bounds are validated by
    `LinkupSearch` at construction.
    """

    def __init__(
        self,
        *,
        client: LinkupClientProtocol | None,
        depth: SearchDepth,
        max_results: int,
        max_text_chars: int,
        include_answer: bool,
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        from_date: date | None = None,
        to_date: date | None = None,
        render_js: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._client = client if client is not None else _default_client()
        self._depth: SearchDepth = depth
        self._max_results = max_results
        self._max_text_chars = max_text_chars
        self._include_domains = list(include_domains) if include_domains else None
        self._exclude_domains = list(exclude_domains) if exclude_domains else None
        self._from_date = from_date
        self._to_date = to_date
        self._render_js = render_js
        self._timeout = timeout
        self.add_function(self.web_search, name="web_search")
        self.add_function(self.get_page, name="get_page")
        if include_answer:
            self.add_function(self.answer, name="answer")

    @_recoverable
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web in real time and return relevant, citable sources with their content.

        Use for current events, facts that may have changed, and anything that needs a
        verifiable source.

        Args:
            query: The search query. Natural-language questions and keyword queries both work.

        Returns:
            The matching pages, each with title, URL, and content.
        """
        try:
            response = await self._client.async_search(
                query,
                depth=self._depth,
                output_type="searchResults",
                from_date=self._from_date,
                to_date=self._to_date,
                exclude_domains=self._exclude_domains,
                include_domains=self._include_domains,
                max_results=self._max_results,
                timeout=self._timeout,
            )
        except LinkupNoResultError:
            return ToolReturn(f"No results found for {query!r}.", metadata={"sources": []})
        results = [r for r in response.results if isinstance(r, LinkupSearchTextResult)]
        results = results[: self._max_results]
        if not results:
            return ToolReturn(f"No results found for {query!r}.", metadata={"sources": []})
        sources = _source_list({result.url: result.name or None for result in results})
        sections = [_format_result(result.name, result.url, result.content) for result in results]
        plural = "s" if len(sections) != 1 else ""
        joined = "\n\n---\n\n".join(sections)
        found = f"Found {len(sections)} result{plural} for {query!r}:\n\n{joined}"
        return ToolReturn(found, metadata={"sources": sources})

    @_recoverable
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Fetch a specific URL and return its content as markdown.

        Use it to read a promising URL from `web_search` results in full, or a URL the user
        provided.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's URL and markdown content.
        """
        response = await self._client.async_fetch(
            url, render_js=self._render_js, timeout=self._timeout
        )
        if not response.markdown:
            raise ModelRetry(
                f"No content could be retrieved for {url!r}. Check the URL or try another page."
            )
        return ToolReturn(
            _format_result(None, url, _truncate(response.markdown, self._max_text_chars)),
            metadata={"sources": _source_list({url: None})},
        )

    @_recoverable
    async def answer(self, question: str) -> ToolReturn[str]:
        """Answer a question from live web sources and return the answer with its sources.

        Suited to direct questions that a single synthesized answer can settle.

        Args:
            question: The question to answer.

        Returns:
            The answer, followed by the sources supporting it.
        """
        response = await self._client.async_search(
            question,
            depth=self._depth,
            output_type="sourcedAnswer",
            from_date=self._from_date,
            to_date=self._to_date,
            exclude_domains=self._exclude_domains,
            include_domains=self._include_domains,
            max_results=self._max_results,
            timeout=self._timeout,
        )
        if not response.answer:
            raise ModelRetry(
                f"Linkup returned no answer for {question!r}. "
                "Rephrase the question, or use web_search."
            )
        sources = {source.url: source.name or None for source in response.sources}
        return ToolReturn(
            _with_sources(response.answer, sources),
            metadata={"sources": _source_list(sources)},
        )


def _format_result(title: str | None, url: str, body: str | None) -> str:
    """Render one result as labelled metadata lines followed by its body text."""
    lines = [] if title is None else [f"Title: {title or '(untitled)'}"]
    lines.append(f"URL: {url}")
    if body:
        lines.extend(["", body])
    return "\n".join(lines)


def _with_sources(body: str, sources: Mapping[str, str | None]) -> str:
    """Append a `Sources:` block to `body`, or return `body` unchanged when there are none."""
    if not sources:
        return body
    lines = [body, "", "Sources:"]
    lines.extend(f"- {title or '(untitled)'}: {url}" for url, title in sources.items())
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    """Cap page text at `max_chars`, keeping the head, where a page's substance usually is.

    The truncation marker counts toward the cap, so the returned text stays within the bound.
    """
    if len(text) <= max_chars:
        return text
    marker = f"\n[... page text truncated at {max_chars} characters]"
    return f"{text[: max(max_chars - len(marker), 0)]}{marker}"
