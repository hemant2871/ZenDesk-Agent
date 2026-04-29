"""
Core agent module — builds and runs the LangChain ReAct agent for
customer support ticket resolution.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from tools.knowledge_base import build_kb_tool, ingest_knowledge_base
from tools.sentiment import build_sentiment_tool
from tools.ticket_tools import build_classify_priority_tool, build_draft_resolution_tool

load_dotenv()


# ---------------------------------------------------------------------------
# ReAct prompt template
# ---------------------------------------------------------------------------

_REACT_TEMPLATE = """\
You are a professional customer support AI agent for Zangoh.
Your job is to resolve customer support tickets accurately, empathetically, and efficiently.

You have access to the following tools:
{tools}

## Resolution Protocol (follow in order):
1. **analyse_sentiment** — Detect the customer's emotional state to guide your tone.
2. **classify_ticket_priority** — Determine how urgent the ticket is.
3. **search_knowledge_base** — Find relevant KB articles that address the issue.
4. **draft_resolution** — Compose and return the final formatted resolution.

## Rules:
- ALWAYS search the knowledge base before answering. Never invent information.
- Match your tone to the sentiment analysis result.
- If no KB article is relevant, say so honestly and suggest escalation to a human agent.
- Do NOT include raw tool outputs in the final answer — use draft_resolution.
- Ticket ID is provided in the human message; always include it in draft_resolution.

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _build_llm() -> ChatGroq:
    """Construct the ChatGroq LLM instance.

    Returns:
        ChatGroq: A llama-3.1-8b-instant model instance.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent_executor() -> AgentExecutor:
    """Build and return a fully initialised LangChain AgentExecutor.

    Ingests the knowledge base into ChromaDB on first run, then wires up
    all tools and the ReAct agent into an AgentExecutor.

    Returns:
        AgentExecutor: Ready-to-invoke agent with all tools attached.

    Raises:
        ValueError: If the Groq API key is missing.
        FileNotFoundError: If the knowledge base JSON is missing.
    """
    # Ensure ChromaDB is populated before the agent starts
    ingest_knowledge_base()

    tools: list[BaseTool] = [
        build_sentiment_tool(),
        build_classify_priority_tool(),
        build_kb_tool(),
        build_draft_resolution_tool(),
    ]

    llm = _build_llm()
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def resolve_ticket(
    executor: AgentExecutor,
    ticket_id: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Run the agent on a single support ticket.

    Args:
        executor: A pre-built AgentExecutor from build_agent_executor().
        ticket_id: The unique identifier for this ticket (e.g. 'TICK-001').
        subject: The subject line of the ticket.
        body: The full message body of the ticket.

    Returns:
        A dict with keys 'output' (str) and 'intermediate_steps' (list).

    Raises:
        Exception: Propagates any unhandled errors from the agent execution.
    """
    user_message = (
        f"Ticket ID: {ticket_id}\n"
        f"Subject: {subject}\n\n"
        f"Customer Message:\n{body}"
    )

    try:
        result: dict[str, Any] = executor.invoke({"input": user_message})
        return result
    except Exception as exc:
        return {
            "output": f"Agent encountered an error: {exc}",
            "intermediate_steps": [],
        }
