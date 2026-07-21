from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from schema import AnswerQuestion, ReviseAnswer  # uses flattened version
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import os


# Load environment variables
load_dotenv(find_dotenv())

# API keys
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Tool output parser (use this for both chains)
tool_output_parser = JsonOutputToolsParser(return_id=True)

# Shared LLM setup
# llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=groq_api_key,
    temperature=0.7,
    max_tokens=512,  # Try lowering this from 1024+
)


def wrap_tool_output(tool_call_dicts):
    if not isinstance(tool_call_dicts, list):
        tool_call_dicts = [tool_call_dicts]

    formatted_tool_calls = []
    for call in tool_call_dicts:
        args = call["args"]

        # ✅ Fix: sanitize references to be only strings
        if "references" in args and isinstance(args["references"], list):
            cleaned_refs = []
            for ref in args["references"]:
                if isinstance(ref, str):
                    cleaned_refs.append(ref)
                elif isinstance(ref, dict):
                    # Grab the key if it's a dict like {"url": "..."} or {"https://...": None}
                    key = next(iter(ref.keys()), None)
                    if key:
                        cleaned_refs.append(key)
            args["references"] = cleaned_refs

        formatted_tool_calls.append({
            "id": call["id"],
            "name": call["type"],
            "args": args
        })

    return AIMessage(content="", tool_calls=formatted_tool_calls)




# === Prompt Template ===
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

# === First Responder ===
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~100 word answer"
)

first_responder_chain = (
    first_responder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | tool_output_parser
    | wrap_tool_output 
)

# === Revisor ===
revise_instructions = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). Format:
    - [1] https://example.com
    - [2] https://example.com
- Remove superfluous information and make sure the revised answer is no more than 50 words.
"""

revisor_chain = (
    actor_prompt_template.partial(first_instruction=revise_instructions)
    | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
    | tool_output_parser
    | wrap_tool_output 
)

# === Sample Test ===
response = first_responder_chain.invoke({
    "messages": [HumanMessage("How can small businesses use AI to grow?")]
})
print(response)
