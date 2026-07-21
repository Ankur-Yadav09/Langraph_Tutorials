import json
from typing import List
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
load_dotenv(find_dotenv())
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Create the Tavily search tool
tavily_tool = TavilySearch(api_key=tavily_api_key, max_results=5)

# Function to execute search queries from tool calls
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            args = tool_call["args"]
            search_queries = args.get("search_queries", [])

            query_results = {}
            for query in search_queries:
                try:
                    result = tavily_tool.invoke(query)
                    short_result = json.dumps(result)[:400]  # Summarize the result in 400 character
                    query_results[query] = short_result
                except Exception as e:
                    query_results[query] = {"error": str(e)}

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )
            )

    return tool_messages


# === Example usage for testing ===
if __name__ == "__main__":
    test_state = [
        HumanMessage(
            content="Write about how small business can leverage AI to grow"
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "",
                        "search_queries": [
                            "AI tools for small business",
                            "AI in small business marketing",
                            "AI automation for small business"
                        ],
                        "reflection_missing": "",
                        "reflection_superfluous": ""
                    },
                    "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                }
            ],
        ),
    ]

    results = execute_tools(test_state)

    print("Raw results:", results)
    if results:
        parsed_content = json.loads(results[0].content)
        print("Parsed content:", json.dumps(parsed_content, indent=2))
