import streamlit as st
import toml
import openai
from langchain.vectorstores.faiss import FAISS
from open_ai_embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import OpenAI
from langchain.docstore.document import Document
from typing import Dict, Any, List


## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.
---------
QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
SOURCES: 1-32
---------
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)


secrets = toml.load('.streamlit/secrets.toml')

openai_api_key = secrets['openai_api_key_azure']
openai.api_key = openai_api_key
openai.api_type = "azure"
openai.api_base = "https://govtext-ds-experiment.openai.azure.com/"
openai.api_version = "2022-12-01"
azure_completion_engine = "text-davinci-003-pretrained"
azure_embedding_engine = "text-embedding-ada-002-pretrained"
chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=openai_api_key, engine=azure_completion_engine), chain_type="stuff", prompt=STUFF_PROMPT)
oai_embedder = OpenAIEmbeddings(query_model_name=azure_embedding_engine, document_model_name=azure_embedding_engine, openai_api_key=openai_api_key)



# @st.cache
# def read_data(allow_output_mutation=True):
#     with open('./govtech_websites_searchsg/docs.pkl', 'rb') as f:
#         doc_chunks = pickle.load(f)

#     db = FAISS.load_local('./govtech_websites_searchsg', oai_embedder)

#     return doc_chunks, db


@st.cache_data
def read_db():
    db = FAISS.load_local('./govtech_websites_searchsg', oai_embedder)
    return db


def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        # original code users "sources", changed to "chunk"
        # print(source_keys)
        # print(doc.metadata["source"])
        if str(doc.metadata["source"]) in source_keys:
            source_docs.append(doc)

    return source_docs


def main():
    st.set_page_config(
        # layout="wide",
        page_title="GovTech Website QA"
        )
    st.title("GovTech Website QA")
    st.caption("Powered by Open AI's GPT-3")
    # st.info("Information taken from [GovText website](https://www.govtext.gov.sg/), [GovText User Guide](https://www.govtext.gov.sg/docs/intro), and this [blob](https://raw.githubusercontent.com/watsonchua/govtext_user_guide_qa/main/docs/govtext/team.txt) about the team members.")


    # doc_chunks, db = read_data() 
    db = read_db()

    sample_questions = [
        "",
        "what does govtech do?",
        "who developed tracetogether?",
        "what are the projects in dsaid?"
       
    ]    

    st.subheader("Question")
    input_container = st.container()
    select_box_question = input_container.selectbox("Examples", sample_questions, key='select_option')
    # input_text = input_container.text_area("Enter Question", key='input_text')

    if select_box_question.strip():
        input_text = input_container.text_area("Enter Question (Type in your own!)", select_box_question, key='input_text')
    else:
        input_text = input_container.text_area("Enter Question (Type in your own!)", key='input_text')

    st.subheader("Answer")
    output_container = st.container()
    sources_container = st.container()


    if input_text.strip():
        with st.spinner():
            query = input_text
            top_docs = db.similarity_search(query, k=2)
            answer =chain({"input_documents": top_docs, "question": query}, return_only_outputs=True)
            answer_sources = get_sources(answer, top_docs)

            output_container.write(answer['output_text'].split('SOURCES:')[0].strip())


            for i,ans_src in enumerate(answer_sources):
                # with st.expander('Source ' + str(i+1)):
                sources_container.subheader('Source ' + str(i+1))
                sources_container.write(ans_src.metadata['url'] + '\n\n' + ans_src.page_content)



if __name__ == "__main__":
    main()