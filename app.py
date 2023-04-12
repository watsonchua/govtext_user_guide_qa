import streamlit as st
import toml
import openai
from prompts import STUFF_PROMPT
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Dict, Any, List
import os
from styling.custom_streamlit_components import format_page_layout
from query_im8 import enquire, format_source, format_content
import json
import datetime
import boto3
from time import perf_counter
# import argparse
import shutil
# openai.api_key = os.environ["OPENAI_API_KEY"]
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

script_run_ctx = get_script_run_ctx()
session_id = script_run_ctx.session_id if script_run_ctx else ''
st.session_state['session_id'] = session_id
# place holder for logging user info
# st.session_state['user'] = 'user'



secrets = toml.load('.streamlit/secrets.toml')

openai_api_key = secrets['openai_api_key_azure']
aws_access_key_id = secrets['aws_access_key_id']
aws_secret_access_key = secrets['aws_secret_access_key']

openai.api_key = openai_api_key

openai.api_type = "azure"
openai.api_base = "https://govtext-ds-experiment.openai.azure.com/"
openai.api_version = "2022-12-01"
azure_embedding_engine = "text-embedding-ada-002"
oai_embedder = OpenAIEmbeddings(query_model_name=azure_embedding_engine, document_model_name=azure_embedding_engine, openai_api_key=openai_api_key, chunk_size=1)

session = boto3.session.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='ap-southeast-1')
s3_client = session.client('s3')

@st.cache_data
def read_db():
    start = perf_counter()
    
    bucket_prefix = 'govtext_user_guide_docs'
    
    target_dir = './' + bucket_prefix
        
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    with open(target_dir + '/index.faiss', 'wb') as f:
        s3_client.download_fileobj('govtext-qa-index-db', bucket_prefix + '/index.faiss', f)
    with open(target_dir + '/index.pkl', 'wb') as f:
        s3_client.download_fileobj('govtext-qa-index-db', bucket_prefix + '/index.pkl', f)
        
    db_clauses = FAISS.load_local(target_dir, oai_embedder)
        
    end = perf_counter()
    print('FAISS index downloaded in ', str(end-start), ' seconds')
    return db_clauses



def log_query(feedback=None):
    if feedback is None:
        log_dict = {'feedback': feedback, **st.session_state}
        prefix = 'all'
    else:
        prefix = feedback
        log_dict = {
            'feedback': feedback,
            'feedback_text': st.session_state['feedback_text'],
            'answer_id': st.session_state['answer_id'],
            'question': st.session_state['question'],
            'answer': st.session_state['answer'],
            'user': st.session_state['user']    

        }
            
    content = json.dumps(log_dict)
    response = s3_client.put_object( 
        Bucket='govtext-user-guide-qa-feedback-data',
        Body=content,
        Key= prefix + '/' + st.session_state['answer_id'] + '.json'
    )



def main():
    format_page_layout(page_title="GovText User Guide Question Answering", layout="wide")
        
    db = read_db()

    sample_questions = [
        "",
        "What is the difference between CTM and LDA?",
        "How to upload dataset for topic modelling?",
        "Which summarization methods does GovText have?",        
        "What data classification can GovText hold?",
        "Who do I contact if I cannot access the website?"
    ]
    
    # st.warning('''
    # This system is an **ALPHA** version.  
    # While we strive to provide accurate and up-to-date information, there may be inaccuracies in the content due to ongoing development and technical limitations.  
    # If you have feedback on the system, please submit it [here](https://form.gov.sg/640ffbdbaed11b0011f4b4bb).
    # ''')
    
    center_container = st.container()
    
    center_container.title("GovText User Guide Question Answering")
    # user_email = center_container.text_input(label='Email (ending with gov.sg)', value=st.session_state['user'] if 'user' in st.session_state else '', key='user')
    

    center_container.subheader("Question")
    
    select_box_question = center_container.selectbox("Examples", sample_questions, key='select_option')

    if select_box_question.strip():
        input_text = center_container.text_area("Enter Question (Type in your own!)", select_box_question, key='input_text')
    else:
        input_text = center_container.text_area("Enter Question (Type in your own!)", key='input_text')

    results_container = center_container.empty()
    
        
    qa_col, _, sources_col = results_container.columns((8,1,8))
    qa_col.subheader("Answer")
    sources_col.subheader("Sources")

    placeholder = qa_col.empty()
    
    
    if input_text.strip():                                                 
        with placeholder:
            with st.spinner('Getting Answers.....'):
                query = input_text
                try:
                    answer, source_docs, selected_docs, top_docs = enquire(query, db, k=6)
                except Exception as e:
                    raise

                qa_col.write(answer.strip())

                # sources_col.markdown('**_IM8 Clauses:_**')

                for sd in source_docs:
                    with sources_col.expander(format_source(sd.metadata)):
                        st.write(format_content(sd.page_content))

                st.session_state['answer'] = answer
                st.session_state['question'] = input_text
                st.session_state['source_docs'] = [vars(sd) for sd in source_docs]
                st.session_state['selected_docs'] = [vars(sed) for sed in selected_docs]
                st.session_state['top_docs'] = [vars(td) for td in top_docs]

                cur_datetime = datetime.datetime.now()
                formatted_datetime  = cur_datetime.strftime("%Y%m%d%H%M%S%f")
                st.session_state['answer_generated_timestamp'] = formatted_datetime

                answer_id = st.session_state['session_id'] + '/' + formatted_datetime
                st.session_state['answer_id'] = answer_id

                # if index_type != 'faq':
                #     log_query(feedback=None)

                #     feedback_placeholder = qa_col.empty()

                #     with feedback_placeholder.form(key='feedback_form'):
                #         feedback_rating = st.radio(label='Optional: Rate this answer!', options=['👍', '👎'], horizontal=True, key='feedback_rating')
                #         feedback_text = st.text_area(label='Give your reason and the correct answer (if applicable).', key='feedback_text')                        
                #         submit_form = st.form_submit_button(label='Submit', on_click=log_and_reset)
                        

# def log_and_reset():
#     if st.session_state['feedback_rating']=='👍':
#         log_query(feedback='good')
#     else: 
#         log_query(feedback='bad')
    
#     st.session_state['select_option'] = ""
#     st.session_state['input_text'] = ""
    
#     print(st.session_state['feedback_rating'])
#     print(st.session_state['feedback_text'])

    

    
if __name__ == "__main__":
    main()

