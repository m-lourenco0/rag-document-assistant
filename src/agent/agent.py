import pprint

from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    """
    The state of our agent.
    'messages' uses 'add_messages' which can intelligently handle RemoveMessage.
    'references' is the list of source documents for the *current* turn only.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    references: List[Dict[str, Any]]


def cleanup_node(state: AgentState) -> dict:
    """
    Clears references and removes old tool-related messages (both the call and the result).
    This keeps the state clean and ensures the message history is always valid.
    """
    print("\n--- CLEANING UP STATE FOR NEW TURN ---")

    messages_to_remove_ids = []
    # Iterate through messages to find tool-related ones to remove
    for i, msg in enumerate(state["messages"]):
        # If we find a ToolMessage, we know we should also remove the AIMessage that preceded it.
        if isinstance(msg, ToolMessage):
            # Add the ToolMessage itself for removal
            messages_to_remove_ids.append(str(msg.id))
            # Find the preceding AIMessage that has the matching tool_call_id
            for j in range(i - 1, -1, -1):
                prev_msg = state["messages"][j]
                if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                    tool_call_ids = [tc["id"] for tc in prev_msg.tool_calls]
                    if msg.tool_call_id in tool_call_ids:
                        messages_to_remove_ids.append(str(prev_msg.id))
                        break

    if messages_to_remove_ids:
        print(
            f"Creating RemoveMessage for {len(messages_to_remove_ids)} old tool-related message(s)."
        )

    # Create a RemoveMessage for each identified message ID
    messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in messages_to_remove_ids]

    return {"messages": messages_to_remove, "references": []}


def call_agent_node(state: AgentState, llm) -> dict:
    """The node that calls the LLM to decide the next step."""
    print("\n--- CALLING AGENT (LLM) ---")
    print(f"Message history has {len(state['messages'])} message(s).")
    response = llm.invoke(state["messages"])

    if response.tool_calls:
        print("Agent decided to call a tool.")
    else:
        print("Agent decided to respond directly.")

    return {"messages": [response]}


def router(state: AgentState) -> str:
    """The router that decides the next step based on the LLM's response."""
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

    workflow.add_node("cleanup", cleanup_node)
    workflow.add_node("agent", lambda state: call_agent_node(state=state, llm=llm))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("cleanup")
    workflow.add_edge("cleanup", "agent")
    workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


def run_chat(new_message: str, thread_id: str, agent_graph) -> dict:
    """
    Runs a conversational turn through the agent graph using the direct .invoke()
    method and constructs the final response data dictionary.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=new_message)]}

    print("\n" + "#" * 80)
    print(f"STARTING AGENT RUN for thread '{thread_id}'")
    print(f"Initial Input: '{new_message}'")
    print("#" * 80)

    final_state = agent_graph.invoke(inputs, config=config)

    print("\n--- AGENT RUN FINISHED ---")

    answer = final_state["messages"][-1].content
    references = final_state.get("references", [])

    print(f"\nExtracted {len(references)} references from the final state.")

    return {"answer": answer, "references": references}
