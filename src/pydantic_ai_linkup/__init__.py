"""Linkup web search, sourced answers and page fetching for Pydantic AI agents."""

from importlib.metadata import PackageNotFoundError, version

from pydantic_ai_linkup._capability import LinkupSearch
from pydantic_ai_linkup._toolset import (
    LinkupClientProtocol,
    LinkupSearchToolset,
    LinkupSource,
    SearchDepth,
)

try:
    __version__ = version("pydantic-ai-linkup")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "LinkupClientProtocol",
    "LinkupSearch",
    "LinkupSearchToolset",
    "LinkupSource",
    "SearchDepth",
    "__version__",
]
