import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from llama_index.core import VectorStoreIndex


def do_rag(query_str: str, index: VectorStoreIndex) -> str:
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # from llama_index.llms.openai import OpenAI
    # from llama_index.core import Settings


    chat_engine = index.as_chat_engine(similarity_top_k=20, chunk_size=100)
    response_3 = chat_engine.query(query_str)
    print(response_3)
    return response_3

    # llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # Settings.llm = llm
    # query_engine = index.as_query_engine(streaming=False, similarity_top_k=4)
    # response = query_engine.query (query_str).response
    # return response

def create_index_from_pdf(pdf_file):
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    with open("uploaded_file.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())

    documents = SimpleDirectoryReader('./').load_data()
    index = VectorStoreIndex.from_documents(documents)
    os.remove("uploaded_file.pdf")

    return index


import streamlit as st

st.title('Question Answering App')

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    index = create_index_from_pdf(uploaded_file)
    query_str = st.text_input('Enter your question')

    if query_str:
        with st.spinner('Wait for it...'):
            response = do_rag(query_str=query_str, index=index)
        # st.success('Done!')

        st.text_area("Response:", value=response, height=300)
