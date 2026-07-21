from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv,find_dotenv
import os

# Load environment variables from the nearest .env file
load_dotenv(find_dotenv())

# Get the Groq and Tavily API keys
groq_api_key = os.getenv("GROQ_API_KEY")

# Create a Groq-based Chat model using the selected model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(result)
