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
    PydanticMultiSelector,LLMSingleSelector, LLMMultiSelector
)
from llama_index.core.response_synthesizers import TreeSummarize

from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.query_pipeline import QueryPipeline

from chat_agent.router import RouterComponent
from llama_index.core.settings import Settings

from chat_agent.agent import create_agent, create_sql_agent, _run_pipeline
import gradio as gr


load_dotenv()

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

CURR_DIR= Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = CURR_DIR / 'config.yaml'
MAX_RETRIES=4

def generate_output(query):
    num_retries = MAX_RETRIES
    while num_retries > 0:
        try:
            response = _run_pipeline(qp, sql_agent, summarizer, query)
            print(response)
            return response
        except Exception as e:
            num_retries-=1
    return ""

config = yaml.full_load(open(CONFIG_PATH))

doc_agents = [create_agent(tool_name, config) for tool_name, config in config['tools'].items()]
sql_agent = create_sql_agent(config['sql_agent'])
choices = [config['description'] for _, config in config['tools'].items()]
choices = choices

selector = LLMMultiSelector.from_defaults()

router_c = RouterComponent(
    selector=selector,
    choices=choices,
    components=doc_agents,
    verbose=False
)   

qp = QueryPipeline(chain=[router_c], verbose=False)

summarizer = TreeSummarize(
    llm=Settings.llm,
    summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)

def chat():
    print("Chatbot: Hello! I'm your NU Soccer Agent! How can I help you?")
    while True:
        # Take user input
        user_input = input("You: ")

        # Exit the chat if the user types 'exit' or 'bye'
        if user_input.lower() in ["exit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Generate and display chatbot response
        response = generate_output(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()