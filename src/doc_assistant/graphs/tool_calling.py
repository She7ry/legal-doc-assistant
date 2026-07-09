"""LangGraph 工具调用 ReAct 循环状态图。

替代 ``ToolCallingChatService.ask()`` 中的手写 for 循环：

    ┌───────┐   有 tool_calls 且    ┌───────┐
    │ model ├── iter < max  ───────►│ tools ├──┐
    └──┬──┬─┘                       └───────┘  │
       │  │                                     │
       │  │  有 tool_calls 且   ┌──────────────┐│
       │  └─ iter >= max  ─────►│ final_answer ├┼─► END
       │                        └──────────────┘│
       │  无 tool_calls                           │
       └─────────────────────────────────────────┴─► END

节点说明：
  model        — 绑定 tools 后调用 LLM
  tools        — 执行 AIMessage 中的 tool_calls（由 service 注入回调）
  final_answer — 达到迭代上限后，不带 tools 再调一次 LLM 生成最终回复
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# 图状态：messages 用 add_messages  reducer 追加；iteration 控制 ReAct 轮数
# ---------------------------------------------------------------------------

class ToolCallingState(TypedDict):
    """LangGraph 工具调用循环在节点间传递的状态。

    messages 通过 add_messages reducer 追加 AIMessage / ToolMessage；
    iteration 每执行一轮 tools 节点 +1，达到 max_iterations 则走 final_answer。
    """

    messages: Annotated[list, add_messages]  # 对话 + tool 结果消息列表
    iteration: int  # 当前已执行的工具轮数
    max_iterations: int  # 上限，超出则走 final_answer 节点


def build_tool_calling_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
):
    """编译工具调用 ReAct 循环为 LangGraph StateGraph。

    Parameters
    ----------
    llm:
        LangChain BaseChatModel.
    tools:
        LangChain tools executed by ToolNode.
    """
    llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

    def call_model(state: ToolCallingState) -> dict[str, Any]:
        """节点 model：带 tools 调用 LLM，可能返回 tool_calls。"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def call_model_final(state: ToolCallingState) -> dict[str, Any]:
        """节点 final_answer：达到迭代上限后，不带 tools 生成最终自然语言回复。"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def count_iteration(state: ToolCallingState) -> dict[str, int]:
        return {"iteration": state["iteration"] + 1}

    def should_continue(state: ToolCallingState) -> str:
        """条件边：无 tool_calls → END；超限 → final_answer；否则 → tools。"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        if state["iteration"] >= state["max_iterations"]:
            return "final_answer"
        return "tools"

    graph = StateGraph(ToolCallingState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("count_iteration", count_iteration)
    graph.add_node("final_answer", call_model_final)

    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        should_continue,
        {"tools": "tools", "final_answer": "final_answer", END: END},
    )
    graph.add_edge("tools", "count_iteration")
    graph.add_edge("count_iteration", "model")
    graph.add_edge("final_answer", END)

    return graph.compile()
