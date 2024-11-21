import warnings
warnings.filterwarnings('ignore')

import yaml 
import os 
import streamlit as st
from pathlib import Path 
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.selectors import (
    PydanticMultiSelector,
)
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import get_response_synthesizer

from chat_agent.agent import create_agent

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o", temperature=0.5)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

CURR_DIR= Path(os.path.dirname(os.path.abspath(__file__)))
print(CURR_DIR)
CONFIG_PATH = CURR_DIR / 'config.yaml'

config = yaml.full_load(open(CONFIG_PATH))
tools = [create_agent(tool_name, config) for tool_name, config in config['tools'].items()]

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=tools,
)

st.set_page_config(page_title="SoccerAgent", page_icon="🤖")
st.title("Soccer Chat Agent")

# Initialize session state for chains and knowledge base
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "AI", "content": "Hello! I'm your LlamaIndex Assistant. How can I assist you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# Function to handle streaming responses
def stream_chat_response(query, history=[]):
    try:
        streaming_response = query_engine.query(query)
        for token in streaming_response.response_gen:
            yield token
    except Exception as e:
        yield f"An error occurred: {str(e)}"

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append({"role": "Human", "content": user_query})

    # Display user's message in chat
    with st.chat_message("Human"):
        st.markdown(user_query)

    response_container = st.empty()  # Placeholder for the bot's response
    response_buffer = ""  # Buffer to accumulate the bot's response

    # Stream bot's response
    with st.spinner("SoccerAgent is processing your query..."):
        for token in stream_chat_response(user_query, st.session_state.chat_history):
            if token.strip() == "":
                continue
            response_buffer += token
            response_container.markdown(f"**SoccerAgent:** {response_buffer}")

    # Handle unsuccessful retrieval
    if response_buffer.strip() == "":
        response_buffer = UNSUCCESSFUL_RETRIEVAL_RESPONSE
        response_container.markdown(f"**SoccerAgent:** {response_buffer}")

    # Append the final response to chat history
    st.session_state.chat_history.append({"role": "AI", "content": response_buffer})