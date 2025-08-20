import os
import streamlit as st
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from datetime import datetime, timezone
from pymongo import MongoClient

# --- LOAD ENVIRONMENT ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
chat_collection = db["chat_history"]

# --- DB FUNCTIONS ---
def save_to_db(question, answer):
    try:
        chat_collection.insert_one({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        print(f"DB Error: {e}")

def load_history(limit=5):
    try:
        history = chat_collection.find().sort("timestamp", -1).limit(limit)
        return [f"Q: {h['question']}\nA: {h['answer']}" for h in reversed(list(history))]
    except Exception as e:
        print(f"DB Error: {e}")
        return []

# --- PROCESS DOCUMENT ---
@st.cache_resource
def process_document(file_bytes: bytes):
    """
    Loads and processes the uploaded PDF file to create a vector store.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        chunks = text_splitter.split_documents(docs)

        embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        return vector_store
    finally:
        os.remove(tmp_file_path)

# --- WORKFLOW SETUP ---
def setup_workflow(vector_store):
    retriever = vector_store.as_retriever()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are an enterprise knowledge base assistant.
            Use ONLY the information from the context below to answer the question.
            If the answer is not contained in the context, respond with:
            "I don't know based on the provided document."
            Keep your answer clear, concise, and professional.
            Context:
            {context}
            Question:
            {question}
            Answer:
            """
    )

    parser = StrOutputParser()
    chain = prompt | model | parser

    class GraphState(TypedDict):
        question: str
        response: str
        documents: List[Document]

    graph = StateGraph(GraphState)

    def retriever_document(state: GraphState) -> GraphState:
        question = state['question']
        documents = retriever.invoke(question)
        return {'documents': documents, 'question': question}

    def generate_response(state: GraphState) -> GraphState:
        question = state['question']
        documents = state['documents']

        chat_history = load_history(limit=5)
        history_text = "\n".join(chat_history)

        context = history_text + "\n\n" + "\n\n".join(doc.page_content for doc in documents)
        try:
            response_text = chain.invoke({
                "question": question,
                "context": context
            })
        except Exception as e:
            response_text = f"⚠️ Error generating response: {e}"

        save_to_db(question, response_text)

        return {'response': response_text, 'question': question, 'documents': documents}

    graph.add_node('retriever_document', retriever_document)
    graph.add_node('generate_response', generate_response)
    graph.add_edge(START, 'retriever_document')
    graph.add_edge('retriever_document', 'generate_response')
    graph.add_edge('generate_response', END)

    return graph.compile()

# --- STREAMLIT FRONTEND ---
st.set_page_config(page_title="Enterprise Chatbot", page_icon="🤖")
st.title("Enterprise Knowledge Base Chatbot")

uploaded_file = st.file_uploader("Upload a PDF document to start chatting...", type="pdf")

if uploaded_file:
    # Process the document
    vector_store = process_document(uploaded_file.getvalue())
    work_flow = setup_workflow(vector_store)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = work_flow.invoke({'question': prompt})
                    assistant_response = response['response']
                except Exception as e:
                    assistant_response = f"⚠️ Workflow error: {e}"
                st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    st.info("Please upload a PDF document to begin.")
