# Import libraries 
from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent, tool
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

# Example - 1
'''
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

# Define Custom Tool (Get System Time)
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Returns the current system date and time formatted according to the given string.
    Default format: YYYY-MM-DD HH:MM:SS
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# Combine all tools (Wrapped Tavily search + custom time tool)
tools = [tavily_tool, get_system_time]

# Create the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True  # Enable detailed logging
)

# Invoke Agent
agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")
'''

'''
#Example-2
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.invoke("Obama's first name?")

print(result)
'''