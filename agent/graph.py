"""
agent/graph.py
LangGraph StateGraph for NeuroChat.

Fixes applied:
- typing.List / Dict used everywhere (LangGraph Pydantic compat)
- try/except on every import that moves between LangChain versions
- memory lives outside the graph (no per-turn reset)
- Optional[List[str]] instead of list[str] | None  (Python 3.8 compat)
- from __future__ import annotations REMOVED — it breaks TypedDict inference
  in some LangGraph versions
"""

from typing import Any, Dict, List, Optional, TypedDict

# ── Imports with version-safe fallbacks ───────────────────────────────────────

# AgentExecutor
try:
    from langchain.agents import AgentExecutor
except ImportError:
    from langchain.agents.agent import AgentExecutor  # type: ignore

# create_openai_tools_agent
try:
    from langchain.agents import create_openai_tools_agent
except ImportError:
    from langchain_core.agents import create_openai_tools_agent  # type: ignore

# Memory — moved to langchain_community in LangChain 0.3+
try:
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )
except ImportError:
    from langchain_community.memory import (  # type: ignore
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.tools import get_tools


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NeuroChat, an advanced conversational AI assistant.

You have:
- Persistent memory of this entire conversation
- Access to tools: web search, calculator, Wikipedia, date/time, and weather

Guidelines:
- Think step-by-step before answering
- When using a tool, briefly state why before calling it
- Summarise tool results clearly — never dump raw output at the user
- Be concise and accurate
- Use markdown for code blocks, lists, and tables where helpful
- If a tool fails, handle it gracefully and answer from your own knowledge
"""


# ── AgentState ────────────────────────────────────────────────────────────────
# IMPORTANT: use typing.List / typing.Dict, NOT built-in list/dict.
# LangGraph uses Pydantic to validate state and cannot infer generics from
# PEP-585 built-in types on Python < 3.12.
class AgentState(TypedDict):
    input: str
    chat_history: List[Dict[str, str]]
    output: str
    tools_used: List[str]


# ── Agent node factory ────────────────────────────────────────────────────────
def make_agent_node(executor: Any, memory: Any):
    """
    Closes over executor + memory so both survive Streamlit reruns.
    The node restores memory from serialised state on each call,
    then saves the new turn back before returning.
    """

    def agent_node(state: AgentState) -> AgentState:
        user_input = state["input"]

        # Restore chat history into memory if it was cleared by a rerun
        stored: List[Dict[str, str]] = state.get("chat_history", [])
        if stored and not memory.chat_memory.messages:
            memory.chat_memory.clear()
            for entry in stored:
                if entry.get("role") == "human":
                    memory.chat_memory.add_user_message(entry["content"])
                else:
                    memory.chat_memory.add_ai_message(entry["content"])

        # Run the agent
        try:
            result = executor.invoke({
                "input": user_input,
                "chat_history": memory.chat_memory.messages,
            })
            output: str = result.get("output", "")
            steps: list = result.get("intermediate_steps", [])
        except Exception as ex:
            output = f"⚠️ Agent error: {ex}"
            steps = []

        # Collect tool names used this turn
        tools_used: List[str] = []
        for action, _ in steps:
            name = getattr(action, "tool", None)
            if name and name not in tools_used:
                tools_used.append(name)

        # Persist this turn into memory
        memory.save_context({"input": user_input}, {"output": output})

        # Serialise updated history back to state (plain dicts — JSON-safe)
        serialised: List[Dict[str, str]] = [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": str(m.content),
            }
            for m in memory.chat_memory.messages
        ]

        return {
            "input": user_input,
            "chat_history": serialised,
            "output": output,
            "tools_used": tools_used,
        }

    return agent_node


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_agent(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    selected_tools: Optional[List[str]] = None,
    memory_type: str = "ConversationBuffer",
    window_k: int = 5,
) -> tuple:
    """
    Build and compile a LangGraph agent.
    Returns (compiled_graph, memory).
    Both are stored in st.session_state so they survive Streamlit reruns.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    # Memory
    if memory_type == "ConversationSummary":
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
        )
    elif memory_type == "ConversationWindow":
        memory = ConversationBufferWindowMemory(
            k=window_k,
            memory_key="chat_history",
            return_messages=True,
        )
    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    # Tools
    tool_objects = get_tools(selected_tools or [])

    # Prompt — agent_scratchpad placeholder is required for OpenAI tools agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tool_objects, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tool_objects,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    # Build LangGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", make_agent_node(executor, memory))
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    return workflow.compile(), memory
