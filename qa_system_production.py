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

# --- LOAD ENVIRONMENT ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')


# --- PROCESS DOCUMENT ---
@st.cache_resource
def process_document(file):
    """
    Loads and processes the uploaded PDF file to create a vector store.
    """
    # Create a temporary file to save the uploaded PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load the PDF from the temporary file path
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        chunks = text_splitter.split_documents(docs)

        # Create the embedding model and vector store
        embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        return vector_store
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)


# --- WORKFLOW SETUP ---
def setup_workflow(vector_store):
    """
    Sets up the LangGraph workflow with the provided vector store.
    """
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

    def retriever_document(state: GraphState)-> GraphState:
        question = state['question']
        documents = retriever.invoke(question)
        return {'documents': documents, 'question': question}

    def generate_response(state: GraphState) -> GraphState:
        question = state['question']
        documents = state['documents']

        # Context is only based on retrieved docs (no DB history now)
        context = "\n\n".join(doc.page_content for doc in documents)
        response_text = chain.invoke({
            "question": question,
            "context": context
        })

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
    # Process the document and get the vector store
    vector_store = process_document(uploaded_file)
    
    # Setup the workflow with the new vector store
    work_flow = setup_workflow(vector_store)

    # Initialize chat history in Streamlit's session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get the model's response using the pre-defined workflow
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = work_flow.invoke({'question': prompt})
                assistant_response = response['response']
                st.markdown(assistant_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    # Prompt the user to upload a file
    st.info("Please upload a PDF document to begin.")
