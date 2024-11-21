import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm 
import sys 
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

tqdm.pandas()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a soccer data assistant that translates a JSON string into clear, detailed, and human-readable text. 
            Include all numerical details and make ensure no data is omitted. 
            Highlight key insights and observations—both strengths and weaknesses—based on the data provided. 
            Make the output engaging and easy to understand
            """,
        ),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def process_item(idx, row, llm_chain):
    try:
        data_str = json.dumps(row.to_dict())
        llm_out = llm_chain.invoke({'input': data_str}).content
        return idx, llm_out, None
    except Exception as e:
        return idx, None, e

if __name__ == "__main__":

    df_path = sys.argv[1]
    file_name = os.path.basename(df_path).split('.')[0]

    df = pd.read_csv(df_path)

    llm_chain = prompt | llm

    failed_indices = []
    parsed_results = [None] * len(df)  

    max_threads = 100
    with ThreadPoolExecutor(max_threads) as executor:

        future_to_index = {
            executor.submit(process_item, idx, df.iloc[idx], llm_chain): idx
            for idx in range(len(df))
        }

        for future in tqdm(as_completed(future_to_index), total=len(df)):
            idx = future_to_index[future]
            try:
                idx, result, error = future.result()
                if error:
                    failed_indices.append(idx)
                parsed_results[idx] = result
            except Exception as e:
                failed_indices.append(idx)

    df['Player Summary'] = parsed_results
    df.to_csv(f"{file_name}-player-summary-stats.csv", index=False)
