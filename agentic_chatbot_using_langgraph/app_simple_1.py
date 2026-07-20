# Import the compiled LangGraph chatbot workflow.
# `chatbot` is the graph that processes user messages and returns AI responses.
from agentic_chatbot_backend_5 import chatbot

# Import LangChain message classes.
# HumanMessage represents the user's input in a format understood by LangGraph/LLMs.
from langchain_core.messages import BaseMessage, HumanMessage

# Import Streamlit for building the chat web application.
import streamlit as st


# ----------------------------------------------------------
# Set the title of the Streamlit application.
# ----------------------------------------------------------
st.title("Agentic Chatbot with LangGraph")


# ----------------------------------------------------------
# Configuration passed to the LangGraph workflow.
#
# thread_id:
#   Identifies a unique conversation thread.
#   All requests with the same thread_id share the same
#   conversation history (memory) if a checkpointer is used.
# ----------------------------------------------------------
CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


# ----------------------------------------------------------
# Streamlit Session State
#
# Session state stores variables between reruns.
#
# Streamlit reruns the entire script every time the user
# interacts with the page. Without session_state, the chat
# history would disappear after every message.
# ----------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# ----------------------------------------------------------
# Display all previous messages stored in session_state.
#
# Each message is rendered as either:
# - User message
# - Assistant message
# ----------------------------------------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


# ----------------------------------------------------------
# Display the chat input box.
#
# Returns:
#   None -> if user hasn't typed anything
#   String -> when user presses Enter
# ----------------------------------------------------------
user_input = st.chat_input("Type here")


# ----------------------------------------------------------
# Runs only when the user submits a message.
# ----------------------------------------------------------
if user_input:

    # ------------------------------------------------------
    # Save the user's message in session_state so that it
    # remains visible even after Streamlit reruns.
    # ------------------------------------------------------
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display the user's message immediately.
    with st.chat_message("user"):
        st.text(user_input)


    # ------------------------------------------------------
    # Generate the assistant's response.
    # ------------------------------------------------------
    with st.chat_message("assistant"):

        # --------------------------------------------------
        # chatbot.stream(...)
        #
        # Sends the user's message to the LangGraph workflow.
        #
        # Input:
        # {
        #     "messages": [HumanMessage(...)]
        # }
        #
        # HumanMessage is required because LangGraph works
        # with LangChain message objects instead of plain text.
        #
        # stream_mode="messages"
        #
        # Instead of waiting for the entire response,
        # the LLM streams small chunks (tokens) one by one.
        # --------------------------------------------------
        ai_message = st.write_stream(

            # ------------------------------------------------
            # Generator Expression
            #
            # chatbot.stream() returns:
            #
            # (message_chunk, metadata)
            #
            # Example:
            #
            # (AIMessageChunk(content="Hel"), {...})
            # (AIMessageChunk(content="lo"), {...})
            #
            # We only need the text content, so we extract:
            #
            # message_chunk.content
            #
            # st.write_stream() automatically displays each
            # chunk as it arrives, creating the typing effect.
            # ------------------------------------------------
            message_chunk.content

            for message_chunk, metadata in chatbot.stream(

                # Initial graph state
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                # Conversation configuration
                config=CONFIG,

                # Stream AI response token-by-token
                stream_mode="messages"
            )
        )


    # ------------------------------------------------------
    # Save the complete AI response in session_state.
    #
    # st.write_stream() returns the fully assembled response
    # after streaming finishes.
    # ------------------------------------------------------
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )