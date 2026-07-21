from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

# Create the LangGraph message graph
graph = MessageGraph()
MAX_ITERATIONS = 2  # stop after 2 rounds of tool usage

# Add nodes
graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

# Add transitions
graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

# Conditional edge to either repeat or END based on tool count
def event_loop(state: List[BaseMessage]) -> str:
    # Limit total number of past messages to avoid hitting Groq's context limit
    recent_state = state[-5:]  # Keep only last 5 messages

    count_tool_visits = sum(isinstance(msg, ToolMessage) for msg in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

# Compile the app
app = graph.compile()

# Optional: visualize flow
print(app.get_graph().draw_mermaid())

# Run the graph
response = app.invoke("Write about how small business can leverage AI to grow")

# Extract final tool call result safely
final_tool_call = response[-1].tool_calls[0]
print("\n✅ Final Answer:\n", final_tool_call["args"]["answer"])
print("\n📦 Full Response:\n", response)
