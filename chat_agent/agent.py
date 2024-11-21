import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import os 
import json 
from typing import List
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
    Document, 
    PromptTemplate,
    SQLDatabase, 
    VectorStoreIndex
)
# put data into sqlite db
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.llms import ChatResponse
import re
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from pathlib import Path 

from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import (
    SentenceSplitter
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from sqlalchemy import (
    create_engine,
    MetaData,)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from llama_index.core.retrievers import SQLRetriever

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
    FnComponent
)
sql_database = None

def create_documents(data, meta_data_keys, corpus_key):
    print("Creating Documents")
    documents = []
    for idx, row in data.iterrows():
        text=row[corpus_key]
        metadata = {key : row[key] for key in meta_data_keys}
        _ = Document(text=text, metadata=metadata)
        documents.append(_)
    return documents


def load_and_create_index(documents, persist_dir):
    splitter = SentenceSplitter()
    player_nodes = splitter.get_nodes_from_documents(documents)
    #assert len(documents) == len(player_nodes)

    if not os.path.exists(persist_dir):
        print("Creating Index")
        faiss_index = faiss.IndexFlatL2(3072)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(player_nodes, storage_context=storage_context, show_progress=True)
        index.storage_context.persist(persist_dir)
    else:
        print("Loading Index")
        vector_store=FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index


def create_agent(agent_name, agent_config):
    print(f"Creating {agent_name} Agent")
    data_path = agent_config['data_path']
    meta_data_keys = agent_config['meta_data_keys']
    index_path = agent_config['index_path']
    agent_description = agent_config['description']
    corpus_key = agent_config['corpus_key']

    data = pd.read_csv(data_path)
    documents = create_documents(data, meta_data_keys, corpus_key)
    index = load_and_create_index(documents=documents, persist_dir=index_path)

    query_engine = index.as_query_engine(streaming=True)
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=(
            agent_description
        ),
    )
    return QP(chain=[query_engine])

# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    global sql_database 
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)



class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )

def _get_tableinfo_with_index(tableinfo_dir, idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()

def create_sql_agent(agent_config):
    global sql_database
    dfs = [pd.read_csv(path) for path in agent_config['data_paths']]

    #engine = create_engine("sqlite:///:memory:")
    engine = create_engine("sqlite:///soccer_data.db")
    metadata_obj = MetaData()
    for idx, df in enumerate(dfs):
        tableinfo = _get_tableinfo_with_index(agent_config['tableinfo_dir'], idx)
        print(f"Creating table: {tableinfo.table_name}")
        create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)

    
    table_infos = []
    for table_file in os.listdir(agent_config['tableinfo_dir']):
        file_path  = os.path.join(agent_config['tableinfo_dir'], table_file)
        data = json.load(open(file_path))
        _ = TableInfo(**data)
        table_infos.append(_)
    
    sql_database = SQLDatabase(engine)

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
    ]  # add a SQLTableSchema for each table

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    sql_retriever = SQLRetriever(sql_database)
    table_parser_component = FnComponent(fn=get_table_context_str)

    sql_parser_component = FnComponent(fn=parse_response_to_sql)

    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
        dialect=engine.dialect.name
    )

    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
        )
    response_synthesis_prompt = PromptTemplate(
        response_synthesis_prompt_str,
    )

    llm = OpenAI(model="gpt-4o")

    qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=False,
    )

    qp.add_chain(["input", "table_retriever", "table_output_parser"])
    qp.add_link("input", "text2sql_prompt", dest_key="query_str")
    qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
    qp.add_chain(
        ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
    )
    qp.add_link(
        "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
    )
    qp.add_link(
        "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
    )
    qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
    qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

    return qp 

def _run_pipeline(qp, sql_agent, summarizer, query):
    try:
        text_response = qp.run(query=query)
        try:
            sql_response = sql_agent.run(query=query)
        except Exception as e:
            print(e)
            sql_response = ""
        summary = summarizer.get_response(query, [str(sql_response), str(text_response)])
        return summary
    except Exception as e:
        raise e

# sql_agent,  summarizer,
def run_query_pipeline(qp, sql_agent,  summarizer, query, num_retries=4):
    while num_retries > 0:
        try:
            response = _run_pipeline(qp, sql_agent, summarizer, query)
            return response
            # response = qp.run(query=query)
            # return str(response)
        except Exception as e:
            num_retries-=1
    return ""

# # Function to handle streaming responses
# def stream_chat_response(query, history=[]):
#     try:
#         streaming_response = st.session_state.query_engine.query(query)
#         if 'response_gen' in streaming_response.__dict__.keys():
#             for token in streaming_response.response_gen:
#                 yield token
#         else:
#             yield streaming_response.response
#     except Exception as e:
#         yield f"An error occurred: {str(e)}"