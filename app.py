import streamlit as st
import toml
import openai
import pickle
import numpy as np
import tiktoken
import pandas as pd


secrets = toml.load('.streamlit/secrets.toml')

openai_api_key = secrets['openai_api_key']


COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

openai.api_key=openai_api_key


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

# f"Context separator contains {separator_len} tokens"

@st.cache
def read_df():
    df = pd.read_csv('./docs/govtext/govtext_content.csv')
    df = df.set_index(["section", "subsection"])

    return df

@st.cache(allow_output_mutation=True)
def read_embeddings():
    with open('./docs/govtext/govtext_content_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"], result["usage"]["prompt_tokens"]

# def load_embeddings(fname: str):
#     with open(fname, 'rb') as f:
#         embeddings = pickle.load(f)
    
#     return embeddings


def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

# contexts: dict[(str, str), np.array])
# returns list[(float, (str, str))]
def order_document_sections_by_query_similarity(query: str, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding, _ = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"



def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings,
    show_prompt: bool = True
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    print(df.head())
    print(query)

    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


def main():
    st.set_page_config(
        # layout="wide",
        page_title="GovText User Guide Query"
        )
    st.title("GovText User Guide Query")
    st.caption("Powered by Open AI's GPT-3")
    st.info("Information taken from [GovText website](https://www.govtext.gov.sg/), [GovText User Guide](https://www.govtext.gov.sg/docs/intro), and this [blob](https://raw.githubusercontent.com/watsonchua/govtext_user_guide_qa/main/docs/govtext/team.txt) about the team members.")


    df = read_df() 
    document_embeddings = read_embeddings()

    sample_questions = [
        "",
        "who is the boss?",
        "how many members are there in govtext?",
        "what is the difference between ctm and lda?",
        "how to upload dataset for topic modelling?",
        "what does govtext do?",
        "what are govtext's features?",
        "who is han jing?",        
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


    if input_text.strip():
        with st.spinner():
            answer = answer_query_with_context(input_text, df, document_embeddings, show_prompt=True)
            output_container.write(answer)


if __name__ == "__main__":
    main()