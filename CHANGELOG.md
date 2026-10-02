# CHANGELOG

## v0.1.0 (unreleased)

### Features

- `LinkupSearch` capability for Pydantic AI agents, with `web_search` (Linkup `searchResults`),
  `get_page` (Linkup `/fetch`, URL to markdown) and the opt-in `answer` tool (Linkup
  `sourcedAnswer`).
- `LinkupSearchToolset`, the underlying `FunctionToolset`, usable on its own via
  `LinkupSearch(...).get_toolset()`.
- Agent spec support: load `LinkupSearch` from YAML/JSON specs via `custom_capability_types`.
