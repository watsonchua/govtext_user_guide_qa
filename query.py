import toml
import openai
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from typing import Dict, Any, List


class QueryEngine:
    def __init__(self):
        secrets = toml.load('.streamlit/secrets.toml')
        config = toml.load('./config.toml')
        self.openai_api_type = config['openai_api_type']
        self.openai_api_base = config['openai_api_base']
        self.openai_api_version = config['openai_api_version']
        self.completion_engine_name = config['completion_engine']
        self.embedding_engine_name = config['embedding_engine']
        self.db_bucket_name = config['db_bucket_prefix']
        self.max_output_tokens = config.get('max_output_tokens', 500)
        self.temperature = config.get('temperature', 0)
        self.top_p = config.get('top_p', 1)
        self.frequency_penalty = config.get('frequency_penalty', 0)
        self.presence_penalty = config.get('presence_penalty', 0)
        self.n = config.get('n', 1)
        self.best_of = config.get('best_of', 1)
        self.chain_type = config.get('chain_type', 'map_reduce')
        self.db_top_k = config.get('db_top_k', '10')

        openai_api_key = secrets['openai_api_key_azure']
        openai.api_key = openai_api_key
        openai.api_type = self.openai_api_type
        openai.api_base = self.openai_api_base
        openai.api_version = self.openai_api_version

        self.embedding = OpenAIEmbeddings(query_model_name=self.embedding_engine_name, document_model_name=self.embedding_engine_name, openai_api_key=openai_api_key, chunk_size=1)
        self.llm = AzureOpenAI(deployment_name=self.completion_engine_name, model_name=self.completion_engine_name, openai_api_key=openai_api_key, temperature=self.temperature, max_tokens=self.max_output_tokens, top_p=self.top_p, frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty, n=self.n, best_of=self.best_of)


        self.db = FAISS.load_local('./' + self.db_bucket_name, self.embedding)


    
    def enquire(self, query: str):
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": self.db_top_k})
        # create a chain to answer questions 
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm, 
            chain_type=self.chain_type,
            retriever=retriever,
            return_source_documents=True
            )
        response = chain({"question": query}, return_only_outputs=True)

        print(response)
        answer_text = response['answer']
        source_ids = [r.strip() for r in response['sources'].split(',')]
        top_docs = response['source_documents']

        answer_sources = [d for d in top_docs if d.metadata['source'].strip() in source_ids]
        
        return answer_text, answer_sources, top_docs
    

def format_source(metadata: Dict[str, Any]) -> str:
    split_no = int(metadata['split_no'].strip())

    if split_no == 0:
        return metadata['filename'].strip().title()
    else:
        return metadata['filename'].strip().title() + '--' + str(split_no + 1)

    
def format_content(content: str) -> str:
    content = re.sub(r'([a-z]{1,3}\))\n', r'\1', content)
    return content


def main():
    qe = QueryEngine()
    print(qe.enquire('What is the purpose of IM8?'))

if __name__ == "__main__":
    main()