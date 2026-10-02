"""End-to-end agent runs with `TestModel` and `FunctionModel`, backed by a fake Linkup client."""

from __future__ import annotations

from linkup import LinkupTooManyRequestsError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_linkup import LinkupSearch
from tests.fakes import (
    FakeLinkupClient,
    fetch_response,
    search_results,
    sourced_answer,
    text_result,
)


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _fake_client() -> FakeLinkupClient:
    return FakeLinkupClient(
        search_response=search_results(
            text_result("https://www.linkup.so", name="Linkup", content="Web search for AI.")
        ),
        answer_response=sourced_answer(
            "Linkup is a web search API.", ("https://www.linkup.so", "Linkup")
        ),
        fetch_response=fetch_response("# Linkup\n\nWeb search for AI."),
    )


async def test_test_model_calls_every_tool() -> None:
    client = _fake_client()
    agent = Agent(TestModel(), capabilities=[LinkupSearch(client=client, include_answer=True)])

    result = await agent.run("Tell me about Linkup.")

    returns = {part.tool_name: part for part in _tool_returns(result.all_messages())}
    assert set(returns) == {"web_search", "get_page", "answer"}
    assert returns["web_search"].metadata == {
        "sources": [{"url": "https://www.linkup.so", "title": "Linkup"}]
    }
    assert "Web search for AI." in returns["get_page"].model_response_str()
    assert "Linkup is a web search API." in returns["answer"].model_response_str()


async def test_instructions_reach_the_model() -> None:
    seen: list[str | None] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model), capabilities=[LinkupSearch(client=_fake_client())])
    await agent.run("hi")

    assert seen[0] is not None
    assert "Linkup search API" in seen[0]


async def test_function_model_searches_then_reads_a_page() -> None:
    client = _fake_client()

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = _tool_returns(messages)
        if not returns:
            return ModelResponse(parts=[ToolCallPart("web_search", {"query": "What is Linkup?"})])
        if len(returns) == 1:
            url = returns[0].metadata["sources"][0]["url"]
            return ModelResponse(parts=[ToolCallPart("get_page", {"url": url})])
        return ModelResponse(parts=[TextPart(f"Read {len(returns)} tool results.")])

    agent = Agent(FunctionModel(model), capabilities=[LinkupSearch(client=client)])
    result = await agent.run("What is Linkup?")

    assert result.output == "Read 2 tool results."
    assert client.search_calls[0]["query"] == "What is Linkup?"
    assert client.fetch_calls[0]["url"] == "https://www.linkup.so"


async def test_rate_limit_is_retried_by_the_model() -> None:
    client = _fake_client()
    client.errors.append(LinkupTooManyRequestsError("slow down"))

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _tool_returns(messages):
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(parts=[ToolCallPart("web_search", {"query": "q"})])

    agent = Agent(FunctionModel(model), capabilities=[LinkupSearch(client=client)])
    result = await agent.run("q")

    assert result.output == "ok"
    assert len(client.search_calls) == 2
    retries = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert len(retries) == 1
    assert "slow down" in retries[0].model_response()
