# Pydantic AI Linkup

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Web search for [Pydantic AI](https://ai.pydantic.dev) agents, backed by the
[Linkup API](https://docs.linkup.so).

`LinkupSearch` is a Pydantic AI [capability](https://ai.pydantic.dev/capabilities/overview/) that
gives an agent real-time web research tools, plus short research guidance for its instructions:

| Tool | Linkup endpoint | Purpose |
|---|---|---|
| `web_search` | `/search` (`searchResults`) | Search the web and return relevant, citable sources with title, URL and content. |
| `get_page` | `/fetch` | Fetch a URL and return its content as markdown. |
| `answer` *(opt-in)* | `/search` (`sourcedAnswer`) | Answer a question from live web sources, with the sources supporting it. |

## Installation

```bash
pip install pydantic-ai-linkup
```

or with `uv`:

```bash
uv add pydantic-ai-linkup
```

## Setup

Get an API key at [app.linkup.so](https://app.linkup.so) and export it:

```bash
export LINKUP_API_KEY='YOUR_LINKUP_API_KEY'
```

To pass the key explicitly instead, give `LinkupSearch` a configured client:

```python
from linkup import LinkupClient
from pydantic_ai_linkup import LinkupSearch

capability = LinkupSearch(client=LinkupClient(api_key="YOUR_LINKUP_API_KEY"))
```

## Quickstart

```python
from pydantic_ai import Agent
from pydantic_ai_linkup import LinkupSearch

agent = Agent("anthropic:claude-sonnet-4-6", capabilities=[LinkupSearch()])

result = agent.run_sync("What are the latest developments in quantum computing? Cite sources.")
print(result.output)
```

Each tool returns a `ToolReturn` whose text is what the model sees and whose
`metadata['sources']` lists the URLs and titles behind it, so you can render citations from
`ToolReturnPart.metadata` without parsing the text.

## Configuration

All options are keyword-only and fixed for the agent's lifetime, so they never appear in the
tool schema the model sees.

```python
from datetime import date

from pydantic_ai_linkup import LinkupSearch

capability = LinkupSearch(
    depth="deep",  # 'flash' | 'fast' | 'standard' (default) | 'deep'
    max_results=5,  # results per web_search call (default 10)
    max_text_chars=20_000,  # cap on get_page markdown (default 10,000)
    include_answer=True,  # also expose the `answer` tool (default False)
    include_domains=["arxiv.org"],  # allowlist
    exclude_domains=[],  # denylist
    from_date=date(2026, 1, 1),  # only results published on or after this date
    to_date=None,  # only results published on or before this date
    render_js=False,  # render JavaScript in get_page
    timeout=120.0,  # per-request timeout in seconds
    guidance=None,  # custom instructions; '' disables them
)
```

- `depth='standard'` suits most queries. `'deep'` runs several search iterations for complex,
  multi-hop questions and can take tens of seconds; `'fast'` and `'flash'` trade coverage for
  latency.
- Rate limits, timeouts and unreachable URLs are raised as `ModelRetry`, so the model can wait,
  rephrase or try another page. Authentication and credit errors propagate and stop the run.

### Using the toolset directly

The tools live in `LinkupSearchToolset`, a regular Pydantic AI `FunctionToolset`. Use it when you
want the tools without the capability's instructions:

```python
from pydantic_ai import Agent
from pydantic_ai_linkup import LinkupSearch

agent = Agent("openai:gpt-5.2", toolsets=[LinkupSearch(max_results=5).get_toolset()])
```

### Agent specs

`LinkupSearch` can be loaded from [agent spec](https://ai.pydantic.dev/agent-spec/) files.
Dates are given as `YYYY-MM-DD` strings:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - LinkupSearch:
      depth: deep
      include_answer: true
      from_date: '2026-01-01'
```

```python
from pydantic_ai import Agent
from pydantic_ai_linkup import LinkupSearch

agent = Agent.from_file("agent.yaml", custom_capability_types=[LinkupSearch])
```

## Development

```bash
make install-dev  # uv sync + pre-commit hooks
make lint         # ruff check, ruff format --check, mypy
make test         # pytest with coverage (no network access)
```

## License

[MIT](LICENSE)
