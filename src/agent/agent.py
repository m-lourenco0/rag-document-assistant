import logging

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

from src.agent.prompts import agent_prompt

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Represents the state of the agent's conversation.

    This TypedDict defines the data structure that is passed between nodes in
    the LangGraph.

    Attributes:
        messages: A list of messages in the conversation. The `add_messages`
            annotated reducer intelligently handles appending new messages
            and processing `RemoveMessage` objects to prune the history.
        references: A list of source documents relevant to the *current* turn's
            response. This is cleared at the start of each new turn.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    references: List[Dict[str, Any]]


def cleanup_node(state: AgentState) -> dict:
    """Clears state from the previous turn before the agent acts."""
    logger.debug("\n--- CLEANING UP STATE FOR NEW TURN ---")

    # Step 1: Create a map of tool_call_id to its AIMessage's id.
    tool_call_map = {}
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_map[tc["id"]] = msg.id

    # Step 2: Find all ToolMessages and their corresponding AIMessages to remove.
    messages_to_remove_ids = set()
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            messages_to_remove_ids.add(str(msg.id))
            if ai_message_id := tool_call_map.get(msg.tool_call_id):
                messages_to_remove_ids.add(str(ai_message_id))

    if messages_to_remove_ids:
        logger.debug(
            f"Creating RemoveMessage for {len(messages_to_remove_ids)} old tool-related message(s)."
        )

    messages_to_remove = [RemoveMessage(id=msg_id) for msg_id in messages_to_remove_ids]
    return {"messages": messages_to_remove, "references": []}


def call_agent_node(state: AgentState, llm) -> dict:
    """Invokes the LLM to get the agent's next action.

    This node formats the current message history into a prompt and passes it
    to the LLM. The LLM's response, which can be a direct answer or a request
    to use tools, is then added to the state.
    """
    logger.debug("\n--- CALLING AGENT (LLM) ---")

    messages_with_prompt = agent_prompt.invoke({"messages": state["messages"]})
    logger.debug(
        f"Message history has {len(state['messages'])} message(s). Sending to LLM."
    )

    response = llm.invoke(messages_with_prompt)

    if response.tool_calls:
        logger.debug("Agent decided to call a tool.")
    else:
        logger.debug("Agent decided to respond directly.")

    return {"messages": [response]}


def router(state: AgentState) -> str:
    """Determines the next step in the graph based on the agent's last message."""
    logger.debug("\n--- ROUTING ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug("Decision: Route to 'tools' node.")
        return "tools"
    else:
        logger.debug("Decision: Route to 'END'.")
        return END


def build_chat_graph(tools: list, llm, checkpointer):
    """Constructs and compiles the conversational agent graph.

    This function defines the agent's workflow using a StateGraph. It wires
    together the nodes and defines the edges that
    control the flow of logic based on the output of the 'agent' node.

    Args:
        tools: A list of tools available to the agent.
        llm: The language model for the agent node.
        checkpointer: The checkpointer for persisting graph state.

    Returns:
        A compiled, runnable LangGraph agent.
    """
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
    """Executes a single conversational turn against the agent graph.

    This function serves as the primary interface for sending a user message
    to the agent. It packages the message, invokes the compiled graph with the
    correct configuration for the given conversation thread, and extracts the
    final answer and any source references from the resulting state.

    Args:
        new_message: The user's input message for this turn.
        thread_id: The unique identifier for the conversation thread.
        agent_graph: The compiled LangGraph agent to run.

    Returns:
        A dictionary containing the agent's final "answer" and a list of "references".
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=new_message)]}

    logger.debug("\n" + "#" * 80)
    logger.debug(f"STARTING AGENT RUN for thread '{thread_id}'")
    logger.debug(f"Initial Input: '{new_message}'")
    logger.debug("#" * 80)

    final_state = agent_graph.invoke(inputs, config=config)

    logger.info("\n--- AGENT RUN FINISHED ---")

    answer = final_state["messages"][-1].content
    references = final_state.get("references", [])

    return {"answer": answer, "references": references}
