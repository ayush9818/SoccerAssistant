import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import os 
from llama_index.core import Document

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage
)
from llama_index.core.node_parser import (
    SentenceSplitter
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import Settings



from llama_index.core.tools import QueryEngineTool


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




    

