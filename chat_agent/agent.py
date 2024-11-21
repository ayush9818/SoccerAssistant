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
from chat_agent.sql_utils import (
    _get_tableinfo_with_index, 
    create_table_from_dataframe,
    TableInfo,
    get_table_context_str,
    parse_response_to_sql
)
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
    return tool 

def create_sql_agent(agent_config):

    dfs = [pd.read_csv(path) for path in agent_config['data_paths']]

    engine = create_engine("sqlite:///:memory:")
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



    

