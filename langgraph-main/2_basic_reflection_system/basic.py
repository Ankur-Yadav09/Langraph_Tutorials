from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage , AIMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain  # Custom chains for content generation and reflection

# Load environment variables from a .env file
load_dotenv()

# Define constants for node names
REFLECT = "reflect"
GENERATE = "generate"

# Initialize the message-based graph
graph = MessageGraph()

# Node function for content generation using a predefined chain
def generate_node(state):
    # print(">> GENERATE NODE RECEIVED:", [m.content for m in state])
    # # Invoke the generation chain with the current message state
    return generation_chain.invoke({
        "messages": state
    })

# Node function for reflecting on the generated content
def reflect_node(messages):
    # print(">> REFLECT NODE RECEIVED:", [m.content for m in messages])
    # Invoke the reflection chain to analyze or improve the message
    response = reflection_chain.invoke({
        "messages": messages
    })
    # Return the reflection as a HumanMessage to be used in the next round
    return [HumanMessage(content=response.content)]

# Add the 'generate' and 'reflect' nodes to the graph
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Set the entry point of the graph to the 'generate' node
graph.set_entry_point(GENERATE)

# Define a conditional function to determine whether to continue or stop the graph execution
def should_continue(state):
    # Stop execution if more than 6 message exchanges have occurred
    if (len(state) > 4):
        return END 
    # Otherwise, go to the reflection step
    return REFLECT

# Add conditional transition from GENERATE node to either END or REFLECT
graph.add_conditional_edges(GENERATE, should_continue)

# Add a fixed transition from REFLECT node back to GENERATE
graph.add_edge(REFLECT, GENERATE)

# Compile the graph into an executable app
app = graph.compile()

# Optional: Visualize the graph using Mermaid and ASCII
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))
## Note
'''
Earlier our model (Gemma2-9B-It) is behaving like a helpful assistant that:
expects structured instructions, like:Topic,Tone,Keywords and Links
So we will pass the user input more specific'''

# Execute the app with an initial human message

response = app.invoke(HumanMessage(content="Write a tweet about how AI Agents taking over content creation"))
# Print the final response
print(response)
