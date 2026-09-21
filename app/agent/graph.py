from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import nodes
from app.agent.tools import TOOLS
from app.config import settings


def build(checkpointer=None):
    graph = StateGraph(nodes.Ticket)
    graph.add_node("triage", nodes.triage)
    graph.add_node("gather", nodes.gather)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("draft", nodes.write_draft)
    graph.add_node("review", nodes.review)
    graph.add_node("execute", nodes.execute)
    graph.add_node("save", nodes.save)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", nodes.after_triage, ["gather", "save"])
    graph.add_conditional_edges("gather", nodes.after_gather, ["tools", "draft"])
    graph.add_edge("tools", "gather")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", nodes.after_review, ["execute", "save"])
    graph.add_edge("execute", "save")
    graph.add_edge("save", END)
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph():
    Path(settings().checkpoints_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(settings().checkpoints_path) as saver:
        yield build(saver)


def config(ticket_id: str) -> dict:
    # тред - номер обращения
    return {"configurable": {"thread_id": ticket_id}, "recursion_limit": 30}
