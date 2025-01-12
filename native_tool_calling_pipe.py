"""
title: Native Tool Calling Pipe
author: Marcel Samyn
author_url: https://samyn.co
git_url: https://github.com/iamarcel/open-webui-utils.git
description: Seamless OpenAI API-native tool calling with streaming and multi-call support
required_open_webui_version: 0.5.0
version: 0.1.0
license: MIT
"""

from abc import ABC, abstractmethod
import inspect
import json
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Any,
    Iterable,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    TypedDict,
    Union,
)
import asyncio
from pydantic import BaseModel, Field
from openai import NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition


class ToolSpecParametersProperty(TypedDict):
    description: str
    type: str
    items: NotRequired[dict[str, str]]
    default: NotRequired[Any]
    enum: NotRequired[list[str]]
    maxItems: NotRequired[int]
    minItems: NotRequired[int]
    prefixItems: NotRequired[list[dict[str, Any]]]


class ToolSpecParameters(TypedDict):
    properties: dict[str, ToolSpecParametersProperty]
    required: NotRequired[list[str]]
    type: str
    additionalProperties: NotRequired[bool]


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: ToolSpecParameters


class ToolCallable(TypedDict):
    toolkit_id: str
    callable: Callable
    spec: ToolSpec
    pydantic_model: NotRequired[BaseModel]
    file_handler: bool
    citation: bool


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class EventEmitterMessageData(TypedDict):
    content: str


class EventEmitterStatusData(TypedDict):
    description: str
    done: Optional[bool]


class EventEmitterStatus(TypedDict):
    type: Literal["status"]
    data: EventEmitterStatusData


class EventEmitterMessage(TypedDict):
    type: Literal["message"]
    data: EventEmitterMessageData


class EventEmitter:
    def __init__(
        self,
        __event_emitter__: Optional[
            Callable[[Mapping[str, Any]], Awaitable[None]]
        ] = None,
    ):
        self.event_emitter = __event_emitter__

    async def emit(
        self, message: Union[EventEmitterMessage, EventEmitterStatus]
    ) -> None:
        if self.event_emitter:
            maybe_future = self.event_emitter(message)
            if asyncio.isfuture(maybe_future) or inspect.isawaitable(maybe_future):
                await maybe_future

    async def status(self, description: str, done: Optional[bool] = None) -> None:
        await self.emit(
            EventEmitterStatus(
                type="status",
                data=EventEmitterStatusData(description=description, done=done),
            )
        )

    async def result(self, summary: str, content: str) -> None:
        await self.emit(
            EventEmitterMessage(
                type="message",
                data=EventEmitterMessageData(
                    content=f"\n<details>\n<summary>{summary}</summary>\n{content}\n</details>",
                ),
            )
        )


class ToolCallResult(BaseModel):
    tool_call: ToolCall
    result: Optional[str] = None
    error: Optional[str] = None

    def to_display(self) -> str:
        if self.error:
            return f"\n\n<details>\n<summary>Error executing {self.tool_call.name}</summary>\n{self.error}\n</details>\n\n"
        return f"\n\n<details>\n<summary>{self.tool_call.name} {self.tool_call.arguments}</summary>\n{json.loads(self.result) if self.result else ''}\n</details>\n\n"


class ToolCallingChunk(BaseModel):
    message: Optional[str] = None
    tool_calls: Optional[Iterable[ToolCall]] = None


class ToolCallingModel(ABC):
    """
    ToolCallingModel is an abstract class that defines the interface for a tool calling model.
    """

    @abstractmethod
    def stream(
        self,
        body: dict,
        __tools__: dict[str, ToolCallable] | None,
    ) -> AsyncIterator[ToolCallingChunk]:
        """
        Takes the request body and optional tools, returning ToolCallingChunks.
        When the chunk contains a message, it's immediately shown to the user.
        Tool calls are collected until the stream ends, and then executed.
        When tools have been executed, this method is called again with the tool results, allowing the model to react to it or call new tools.
        """
        raise NotImplementedError

    @abstractmethod
    def append_tool_calls(self, body: dict, tool_calls: Iterable[ToolCall]) -> None:
        """
        Append tool calls to the request body.
        """
        raise NotImplementedError

    @abstractmethod
    def append_results(self, body: dict, results: Iterable[ToolCallResult]) -> None:
        """
        Append the results of tool calls to the request body.
        """
        raise NotImplementedError


class OpenAIToolCallingModel(ToolCallingModel):
    def __init__(self, client: OpenAI, model_id: str):
        self.client = client
        self.model_id = model_id

    async def stream(
        self,
        body: dict,
        __tools__: dict[str, ToolCallable] | None,
    ) -> AsyncIterator[ToolCallingChunk]:
        tools = self._map_tools(__tools__)
        messages: list[ChatCompletionMessageParam] = body["messages"]

        tool_calls_map: dict[str, ToolCall] = {}
        last_tool_call_id: Optional[str] = None

        for chunk in self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
            tools=tools or NotGiven(),
        ):
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                yield ToolCallingChunk(message=delta.content)

            for tool_call in delta.tool_calls or []:
                # Tool call id is only given when the block starts.
                # Keep track of it as function name and arguments come in in later chunks.
                tool_call_id = tool_call.id or last_tool_call_id
                last_tool_call_id = tool_call_id

                if not tool_call_id:
                    continue

                if tool_call_id not in tool_calls_map:
                    tool_calls_map[tool_call_id] = ToolCall(
                        id=tool_call_id, name="", arguments=""
                    )

                if tool_call.function:
                    if tool_call.function.name:
                        tool_calls_map[tool_call_id].name = tool_call.function.name
                    if tool_call.function.arguments:
                        tool_calls_map[
                            tool_call_id
                        ].arguments += tool_call.function.arguments

            if finish_reason:
                if tool_calls_map:
                    yield ToolCallingChunk(tool_calls=tool_calls_map.values())
                return

    def append_results(self, body: dict, results: Iterable[ToolCallResult]):
        if "messages" in body:
            for result in results:
                body["messages"].append(self._map_result(result))

    def append_tool_calls(self, body: dict, tool_calls: Iterable[ToolCall]):
        if "messages" in body:
            body["messages"].append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

    def _map_result(self, result: ToolCallResult) -> dict[str, str]:
        if result.error:
            return {
                "role": "tool",
                "tool_call_id": result.tool_call.id,
                "content": result.error,
            }
        return {
            "role": "tool",
            "tool_call_id": result.tool_call.id,
            "content": result.result or "",
        }

    def _map_tools(
        self, tool_specs: dict[str, ToolCallable] | None
    ) -> list[ChatCompletionToolParam]:
        openai_tools: list[ChatCompletionToolParam] = []
        for tool in tool_specs.values() if tool_specs else []:
            function_definition: FunctionDefinition = {
                "name": tool["spec"]["name"],
                "description": tool["spec"].get("description"),
                "parameters": tool["spec"].get("parameters"),  # type: ignore
            }
            openai_tools.append(
                {
                    "type": "function",
                    "function": function_definition,
                }
            )
        return openai_tools


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1", description="OpenAI API base URL"
        )
        MODEL_IDS: list[str] = Field(
            default=["gpt-4o-mini"],
            description="List of model IDs to enable (comma-separated)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.type = "manifold"
        self.name = "native-tool/"

    def pipes(self) -> list[dict]:
        return [
            {"id": model_id, "name": model_id} for model_id in self.valves.MODEL_IDS
        ]

    async def execute_tool(
        self,
        tool_call: ToolCall,
        tools: dict[str, ToolCallable],
        ev: EventEmitter,
    ) -> ToolCallResult:
        try:
            tool = tools.get(tool_call.name)
            if not tool:
                raise ValueError(f"Tool '{tool_call.name}' not found")

            parsed_args = json.loads(tool_call.arguments)
            await ev.status(
                f"Executing tool '{tool_call.name}' with arguments: {parsed_args}"
            )

            result = await tool["callable"](**parsed_args)

            return ToolCallResult(
                tool_call=tool_call,
                result=json.dumps(result),
            )
        except json.JSONDecodeError:
            return ToolCallResult(
                tool_call=tool_call,
                error=f"Failed to parse arguments for tool '{tool_call.name}'",
            )
        except Exception as e:
            return ToolCallResult(
                tool_call=tool_call,
                error=f"Error executing tool '{tool_call.name}': {str(e)}",
            )

    async def pipe(
        self,
        body: dict,
        __user__: dict | None = None,
        __task__: str | None = None,
        __tools__: dict[str, ToolCallable] | None = None,
        __event_emitter__: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[str, None]:
        if __task__ == "function_calling":
            # Go away open-webui let me deal with it myself
            return

        client = OpenAI(
            api_key=self.valves.OPENAI_API_KEY, base_url=self.valves.OPENAI_BASE_URL
        )

        model_id = body["model"] or ""
        model_id = model_id[model_id.find(".") + 1 :]

        model = OpenAIToolCallingModel(client, model_id)
        ev = EventEmitter(__event_emitter__)

        while True:
            await ev.status("Generating response...")
            tool_calls: list[ToolCall] = []

            # Stream model response: pass text content through and collect tool calls
            async for chunk in model.stream(body, __tools__):
                tool_calls = list(chunk.tool_calls) if chunk.tool_calls else tool_calls

                if chunk.message:
                    yield chunk.message

            if not tool_calls:
                # No tools to execute, stop the loop
                await ev.status("Done", done=True)
                break

            if not __tools__:
                raise ValueError("No tools provided while tool call was requested")

            model.append_tool_calls(body, tool_calls)

            # Execute tools and process results
            await ev.status("Executing tools...")
            tool_call_results = [
                await self.execute_tool(
                    tool_call,
                    __tools__,
                    ev,
                )
                for tool_call in tool_calls
            ]

            # Add to body for next iteration(s)
            model.append_results(body, tool_call_results)

            # Yield result for later conversation turns
            for result in tool_call_results:
                yield result.to_display()

            tool_calls = []
            await ev.status("Tool execution complete", done=True)

        return
