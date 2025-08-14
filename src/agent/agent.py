import operator

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    references: Annotated[List[Dict[str, Any]], operator.add]


def call_agent_node(state: AgentState, llm) -> dict:
    print("\n--- CALLING AGENT (LLM) ---")
    print(f"Message history has {len(state['messages'])} message(s).")
    response = llm.invoke(state["messages"])

    if response.tool_calls:
        print("Agent decided to call a tool.")
    else:
        print("Agent decided to respond directly.")

    return {"messages": [response]}


def router(state: AgentState) -> str:
    print("\n--- ROUTING ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("Decision: Route to 'tools' node.")
        return "tools"
    else:
        print("Decision: Route to 'END'.")
        return END


def build_chat_graph(tools: list, llm, checkpointer):
    """Builds and compiles the agentic chat graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", lambda state: call_agent_node(state=state, llm=llm))
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


def run_chat(new_message: str, thread_id: str, agent_graph) -> dict:
    """
    Runs a conversational turn through the agent graph and constructs the
    final response data dictionary, printing each state change.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=new_message)]}

    print("\n" + "#" * 80)
    print(f"STARTING AGENT RUN for thread '{thread_id}'")
    print(f"Initial Input: '{new_message}'")
    print("#" * 80)

    # This dictionary will accumulate the full state as the graph runs
    full_state = {}

    for event in agent_graph.stream(inputs, config=config):
        # The event is a dictionary with the node name as the key
        full_state.update(event)

    print("\n--- AGENT RUN FINISHED ---")
    response_state = full_state.get("agent", {})
    answer = response_state["messages"][-1].content
    references = []
    for doc in full_state.get("references", []):
        references.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("filename", "N/A"),
                "page": doc.metadata.get("page_number", "N/A"),
            }
        )

    return {"answer": answer, "references": references}
