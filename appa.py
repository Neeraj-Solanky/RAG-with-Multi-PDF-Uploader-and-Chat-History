import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure NVIDIA API key is set
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    """
    This function creates embeddings for documents loaded from the PDF and stores them in session state.
    """
    if "vectors" not in st.session_state:
        # Initializing Nvidia embeddings and loading the documents
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("genai_principles.pdf")
        st.session_state.docs = st.session_state.loader.load()

        # Splitting the loaded documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

        # Storing the vectors in session state using FAISS for retrieval
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        print("Embeddings and vector store are ready.")

# Streamlit UI Title
st.title("Nvidia NIM Demo")

# Initialize LLM with Nvidia's Chat model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Creating a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Input field for the user to ask questions
prompt1 = st.text_input("Enter Your Question from Documents")

# Button to start embedding documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready.")

# If a question is provided, retrieve and respond
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Track time for performance
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")

    # Display the answer
    st.write(response['answer'])

    # Expand section for displaying relevant document chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

