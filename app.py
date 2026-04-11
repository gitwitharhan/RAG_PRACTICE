__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c 0%, #0d0e14 100%);
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(79, 172, 254, 0.05);
        border: 1px solid rgba(79, 172, 254, 0.2);
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatMistralAI(model="mistral-small-2603")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful AI assistant.
        Use ONLY the provided context text to answer the question.
        If the answer is not present in the context, say "I could not find the answer in the uploaded document."
        """),
        ("human", """
        Context: {context}
        Question: {question}
        """)
    ])
    return embeddings, llm, prompt_template

embeddings_model, llm_model, rag_prompt = load_rag_components()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.markdown("## 📄 Document Center")
    uploaded_file = st.file_uploader("Upload PDF Knowledge Base", type="pdf")
    
    if uploaded_file:
        with st.status("🚀 Processing Knowledge Base...", expanded=True) as status:
            st.write("Reading file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.write("Splitting text into chunks...")
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            st.write("Generating embeddings & Indexing...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
            )
            
            st.session_state.retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5}
            )
            
            os.remove(tmp_path)
            status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
            st.success("Knowledge Base successfully indexed!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<h1 class="main-title">Semantic RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #888;'>Upload a PDF to start a semantic conversation with your data.</p>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if st.session_state.retriever:
            with st.spinner("Analyzing document..."):
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                final_prompt = rag_prompt.invoke({"context": context, "question": query})
                response = llm_model.invoke(final_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            st.warning("⚠️ Please upload a PDF in the sidebar first to enable the RAG system.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF to proceed."})
