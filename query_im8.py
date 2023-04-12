from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from prompts import STUFF_PROMPT, template
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import OpenAI
from langchain.vectorstores.faiss import FAISS
from typing import Dict, Any, List
import toml
import openai
import re
import tiktoken
from prompts import azure_template

secrets = toml.load('.streamlit/secrets.toml')

openai_api_key = secrets['openai_api_key_azure']
openai.api_key = openai_api_key
openai.api_type = "azure"
openai.api_base = "https://govtext-ds-experiment.openai.azure.com/"
openai.api_version = "2022-12-01"
# azure_completion_engine = "text-davinci-003-pretrained"
# azure_completion_engine = "gpt-35-turbo-0301-pretrained"
# azure_completion_engine = "gpt-35-turbo"
azure_completion_engine = "text-davinci-003"
azure_embedding_engine = "text-embedding-ada-002"

MAX_OUTPUT_TOKENS = 500
# embedding model
oai_embedder = OpenAIEmbeddings(query_model_name=azure_embedding_engine, document_model_name=azure_embedding_engine, openai_api_key=openai_api_key, chunk_size=1)
chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=openai_api_key, engine=azure_completion_engine, max_tokens=MAX_OUTPUT_TOKENS), chain_type="stuff", prompt=STUFF_PROMPT)

# from langchain.chat_models import AzureChatOpenAI
# chat = AzureChatOpenAI(temperature=0, openai_api_key=openai_api_key, deployment_name="gpt-35-turbo", openai_api_base="https://govtext-ds-experiment.openai.azure.com/", openai_api_version="2023-03-15-preview") #TODO: enter openai_api_base using named parameter after langchain fix
# print(chat)
# chain = load_qa_with_sources_chain(llm=chat, chain_type="stuff", prompt=STUFF_PROMPT)

def max_tokens_for_context(question, model="text-davinci-003", max_output_tokens=500):
    encoding = tiktoken.encoding_for_model(model)
    source_buffer = 200
    num_tokens = len(encoding.encode(template)) + len(encoding.encode(question)) + max_output_tokens + source_buffer
    return 4095 - num_tokens
    
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""
    # Get sources for the answer
    source_info = answer["output_text"].split("SOURCES:")
    source_entry = source_info[-1].strip()
    if source_entry.endswith('.'):
        source_entry = source_entry[:-1]
    answer_text = source_info[0].strip()
    
    source_keys = [] if not source_entry else [s.strip() for s in source_entry.split(",")]
    
    source_docs = []
    for doc in docs:
        if str(doc.metadata["source"].strip()) in source_keys:
            source_docs.append(doc)

    return answer_text, source_docs


def format_source(metadata: Dict[str, Any]) -> str:
    split_no = int(metadata['split_no'].strip())

    if split_no == 0:
        return metadata['filename'].strip().title()
    else:
        return metadata['filename'].strip().title() + '--' + str(split_no + 1)

    
def format_content(content: str) -> str:
    content = re.sub(r'([a-z]{1,3}\))\n', r'\1', content)
    return content

def enquire(query: str, faiss_db:FAISS, k:int=10):
    max_context_tokens = max_tokens_for_context(query, model="text-davinci-003", max_output_tokens=MAX_OUTPUT_TOKENS)
    top_docs = faiss_db.similarity_search(query, k=k)
    selected_docs = []
    cur_length = 0
    for t in top_docs:
        new_length = cur_length + t.metadata['length']
        if new_length <= max_context_tokens:
            selected_docs.append(t)
            cur_length = new_length
            
    answer = chain({"input_documents": top_docs, "question": query}, return_only_outputs=True)
    answer_text, answer_sources = get_sources(answer, selected_docs)
    
    return answer_text, answer_sources, selected_docs, top_docs

    
    

