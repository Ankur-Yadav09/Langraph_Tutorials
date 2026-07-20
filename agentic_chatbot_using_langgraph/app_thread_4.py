from agentic_chatbot_backend_5 import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid

# ==========================================================
# Streamlit + LangGraph Chatbot
#
# Workflow:
# 1. Initialize Streamlit session state.
# 2. Generate a unique thread ID for each conversation.
# 3. Display all previous conversations in the sidebar.
# 4. User can create a new chat or switch to an old one.
# 5. User sends a message.
# 6. Message is sent to LangGraph.
# 7. LangGraph streams the AI response.
# 8. Response is displayed and stored in session state.
#
# Note:
# Streamlit reruns this entire script whenever the user
# interacts with the page. Session state is used to
# preserve data across reruns.
# ==========================================================


# ==========================================================
# Helper Functions
# These functions manage conversation threads and retrieve
# conversation history from the LangGraph checkpointer.
# ==========================================================

# Generate a unique thread ID for every new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new conversation thread to the sidebar list
def add_thread(thread_id):

    # Avoid duplicate thread IDs
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new conversation
def reset_chat():

    # Generate a new thread for LangGraph memory
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the UI messages
    st.session_state["message_history"] = []

    # Add this conversation to the sidebar
    add_thread(st.session_state["thread_id"])


# Load conversation history from the LangGraph checkpointer
def load_conversation(thread_id):

    # Retrieve graph state using thread_id
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return all stored messages
    return state.values.get("messages", [])


# ==========================================================
# Streamlit UI Initialization
# ==========================================================

st.title("Agentic Chatbot with LangGraph")


# ==========================================================
# Initialize Session State
#
# Session State stores:
# 1. message_history -> UI chat history
# 2. thread_id       -> Current LangGraph conversation
# 3. chat_threads    -> List of all conversations
# ==========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# Register the current conversation
add_thread(st.session_state["thread_id"])


# ==========================================================
# Sidebar
#
# Allows the user to:
# • Create a new conversation
# • Switch between previous conversations
# ==========================================================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):

    # Start a fresh conversation
    reset_chat()

    # Restart the script to refresh the UI
    st.rerun()


# ==========================================================
# Display all saved conversations
#
# Newest conversations appear at the top.
# ==========================================================

for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(str(thread_id), key=thread_id):

        # Switch to the selected conversation
        st.session_state["thread_id"] = thread_id

        # Load conversation from LangGraph memory
        messages = load_conversation(thread_id)

        # Convert LangChain messages into a format
        # that Streamlit understands.
        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore ToolMessage or other message types
            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history
        st.session_state["message_history"] = temp_messages

        # Refresh the page
        st.rerun()


# ==========================================================
# Main Chat Interface
#
# Display all messages stored in the current conversation.
# ==========================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.text(message["content"])


# ==========================================================
# Process User Input
#
# This block runs only after the user submits a message.
# ==========================================================

user_input = st.chat_input("Type here")

if user_input:

    # Save the user message in the UI memory
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user message immediately
    with st.chat_message("user"):
        st.text(user_input)

    # Configure LangGraph to use the current conversation.
    # Every request with the same thread_id shares the same
    # conversation memory inside the graph.
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # ======================================================
    # Conversation Flow
    #
    # User
    #   │
    #   ▼
    # HumanMessage
    #   │
    #   ▼
    # LangGraph Workflow
    #   │
    #   ▼
    # LLM
    #   │
    #   ▼
    # AIMessage Chunks
    #   │
    #   ▼
    # st.write_stream()
    # ======================================================

    with st.chat_message("assistant"):

        # chatbot.stream() streams the response token-by-token.
        # st.write_stream() displays each token immediately,
        # producing a ChatGPT-like typing effect.
        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata in chatbot.stream(
                {
                    # Only the latest user message is passed.
                    # Previous messages are automatically loaded
                    # by LangGraph using the thread_id.
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=CONFIG,

                stream_mode="messages"
            )

            # Ignore HumanMessage and ToolMessage chunks.
            # Display only AI-generated text.
            if isinstance(message_chunk, AIMessage)
        )

    # Save the completed AI response so it remains visible
    # after the next Streamlit rerun.
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })