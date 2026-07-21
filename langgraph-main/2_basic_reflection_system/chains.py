from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import os
# from langchain_openai import ChatOpenAI

# Load environment variables from the nearest .env file
load_dotenv(find_dotenv())

# Get the Groq and Tavily API keys
groq_api_key = os.getenv("GROQ_API_KEY")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create a Groq-based Chat model using the selected model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
# llm = ChatOpenAI(model="gpt-4o")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
