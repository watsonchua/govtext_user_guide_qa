from open_ai_embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from prompts import STUFF_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import OpenAI
from langchain.vectorstores.faiss import FAISS
from typing import Dict, Any, List
import toml
import openai
import re
import argparse
from azure_chatgpt_query import AzureChatGPT
import tiktoken
from prompts import azure_template

secrets = toml.load('.streamlit/secrets.toml')

openai_api_key = secrets['openai_api_key_azure']
openai.api_key = openai_api_key
openai.api_type = "azure"
openai.api_base = "https://govtext-ds-experiment.openai.azure.com/"
openai.api_version = "2022-12-01"
# azure_completion_engine = "text-davinci-003-pretrained"
azure_completion_engine = "gpt-35-turbo-0301-pretrained"
azure_embedding_engine = "text-embedding-ada-002-pretrained"

# embedding model
oai_embedder = OpenAIEmbeddings(query_model_name=azure_embedding_engine, document_model_name=azure_embedding_engine, openai_api_key=openai_api_key)

generation_params = {
    'temperature':0,
    'max_tokens':500,
    'top_p':1,
    'frequency_penalty':0,
    'presence_penalty':0,
    'best_of':1,
    'stop':["<|im_end|>"]
}

azure_chat_gpt = AzureChatGPT(engine_name=azure_completion_engine, **generation_params)
# chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=openai_api_key, engine=azure_completion_engine), chain_type="stuff", prompt=STUFF_PROMPT)


def max_tokens_for_context(question, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    source_buffer = 200
    num_tokens = len(encoding.encode(azure_template)) + len(encoding.encode(question)) + generation_params['max_tokens'] + source_buffer
    return 4095 - num_tokens
    



def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""
    # Get sources for the answer
    # print(answer["output_text"])
    source_info = answer["output_text"].split("SOURCES:")
    source_entry = source_info[-1].strip()
    if source_entry.endswith('.'):
        source_entry = source_entry[:-1]
    answer_text = source_info[0].strip()
    
    source_keys = [] if not source_entry else [s.strip() for s in source_entry.split(",")]
    # print(source_keys)
    # print(len(source_keys))
    
    source_docs = []
    for doc in docs:
        if str(doc.metadata["source"].strip()) in source_keys:
            source_docs.append(doc)

    # print([s.metadata['source'] for s in source_docs])
    # print(len(source_docs))
    return answer_text, source_docs


def format_source(metadata: Dict[str, Any]) -> str:
    # src_tokens = src.split('/', maxsplit=1) # to set back to 1 after removing root folder path from source
    # src_tokens = src.split('/', maxsplit=2)
    # formatted = re.sub(r'--\d$', '', src_tokens[-1])
    # if metadata['clause_no'].strip():
    #     formatted = metadata['filename'].strip() + '--' + metadata['clause_no'].strip()
    # else:
    split_no = int(metadata['split_no'].strip())

    if split_no == 0:
        return metadata['filename'].strip().title()
    else:
        return metadata['filename'].strip().title() + '--' + str(split_no + 1)

    
def format_content(content: str) -> str:
    content = re.sub(r'([a-z]{1,3}\))\n', r'\1', content)
    return content

def enquire(query: str, faiss_db:FAISS, k:int=10):
    max_context_tokens = max_tokens_for_context(query, model="gpt-3.5-turbo-0301")
    top_docs = faiss_db.similarity_search(query, k=k)
    selected_docs = []
    cur_length = 0
    for t in top_docs:
        new_length = cur_length + t.metadata['length']
        if new_length <= max_context_tokens:
            selected_docs.append(t)
            cur_length = new_length
            
    # answer = chain({"input_documents": top_docs, "question": query}, return_only_outputs=True)
    answer = azure_chat_gpt.complete(query, selected_docs)
    answer_text, answer_sources = get_sources(answer, selected_docs)
    
    return answer_text, answer_sources, selected_docs, top_docs


def enquire_multiple(query: str, db_main:FAISS, db_secondary:FAISS, k:int=10):
    max_context_tokens = max_tokens_for_context(query, model="gpt-3.5-turbo-0301")
    top_docs_main = db_main.similarity_search(query, k=k)
    top_docs_secondary = db_secondary.similarity_search(query, k=k)
    top_docs = top_docs_main + top_docs_secondary
    selected_docs = []
    cur_length = 0
    for t in top_docs:
        new_length = cur_length + t.metadata['length']
        if new_length <= max_context_tokens:
            selected_docs.append(t)
            cur_length = new_length
            
    # answer = chain({"input_documents": top_docs, "question": query}, return_only_outputs=True)
    answer = azure_chat_gpt.complete(query, selected_docs)
    answer_text, answer_sources = get_sources(answer, selected_docs)
    
    return answer_text, answer_sources, selected_docs, top_docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', help='query string') 
    parser.add_argument('-f', '--faiss', help='faiss db folder')
    args = parser.parse_args()
    
    db = FAISS.load_local(args.faiss, oai_embedder)

    # db = FAISS.load_local("./im8_docs_parsed_by_clauses", oai_embedder)
    answer_text, source_docs, selected_docs, top_docs = enquire(args.query, db)
    
    print('Answer:', answer_text)
    for doc in source_docs:
        print('------------------------------------------------')
        print('Source:', format_source(doc.metadata))
        print('------------------------------------------------')
        print(format_content(doc.page_content))
        
        

if __name__ == "__main__":
    main()
    
    

