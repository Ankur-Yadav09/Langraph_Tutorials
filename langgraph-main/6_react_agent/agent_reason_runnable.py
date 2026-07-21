from langchain.agents import tool, create_react_agent
import datetime
from langchain import hub
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
import datetime
import os
from langchain.tools import Tool
from langchain_tavily import TavilySearch

# Load environment variables from the nearest .env file
load_dotenv(find_dotenv())

# Get the Groq and Tavily API keys
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Create a Groq-based Chat model using the selected model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initialize the TavilySearch object
tavily = TavilySearch(api_key=tavily_api_key)

# Wrap the Tavily tool for single input compatibility
def tavily_tool_wrapper(query: str) -> str:
    return tavily.invoke(query)

# Register wrapped Tavily tool
tavily_tool = Tool(
    name="tavily_search",
    func=tavily_tool_wrapper,
    description="Search the web for up-to-date information using a search query."
)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

react_prompt = hub.pull("hwchase17/react")

tools = [get_system_time, tavily_tool]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)