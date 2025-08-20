import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

import datetime
from pymongo import MongoClient

import streamlit as st

import asyncio

# Fix for Streamlit "no current event loop" issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    

# load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')


mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
chat_collection = db["chat_history"]

def save_to_db(question, answer):
    chat_collection.insert_one({
        "question": question,
        "answer": answer,
        # "timestamp": datetime.datetime.utcnow()
        "timestamp": datetime.datetime.now(datetime.UTC)
    })

def load_history(limit=5):
    history = chat_collection.find().sort("timestamp", -1).limit(limit)
    return [f"Q: {h['question']}\nA: {h['answer']}" for h in reversed(list(history))]

# for all type of data formate
# def load_document(file_path):
#     loader = UnstructuredFileLoader(file_path)
#     docs = loader.load()
#     return docs

# docs = load_document("instruction.docx")
# print(len(docs))


# loader = PyPDFLoader("instruction2.pdf")
# docs = loader.load()
# # print(len(docs))

# text_spliter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=20
# )

# chunks = text_spliter.split_documents(docs)
# # print(len(chunks))

# embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
# # print(vector_store)

# retriever = vector_store.as_retriever()

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
    
    chat_history = load_history(limit=5)
    history_text = "\n".join(chat_history)
    
    context = history_text+"\n\n".join(doc.page_content for doc in documents)
    generate_response = chain.invoke({
        "question": question,
        "context": context
    })
    
    save_to_db(question, generate_response)
    
    return {'response': generate_response, 'question': question, 'documents': documents}
    

graph.add_node('retriever_document', retriever_document)
graph.add_node('generate_response', generate_response)

graph.add_edge(START, 'retriever_document')
graph.add_edge('retriever_document', 'generate_response')
graph.add_edge('generate_response', END)


work_flow = graph.compile()

# initial_state = {'question': 'What is the company’s leave policy?'}
# final_flow = work_flow.invoke(initial_state)
# print(final_flow['response'])

# while True:
#     user_input = input("You:")
#     if user_input.lower() in ["exit", "bye"]:
#         print("Come back Soon :)")
#         break
#     else:
#         response= work_flow.invoke({'question': user_input})
#         print(response['response'])




# --------------------------
# STREAMLIT FRONTEND
# --------------------------
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == "__main__":
    st.set_page_config(page_title="Enterprise Q&A Chatbot", layout="wide")
    st.title("📘 Enterprise Knowledge Base Chatbot")

    # Sidebar for PDF upload
    st.sidebar.header("Upload a PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

    retriever = None  # Initialize retriever
    if uploaded_file is not None:
        with open("user_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("user_uploaded.pdf")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = text_splitter.split_documents(docs)

        embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        retriever = vector_store.as_retriever()
        st.sidebar.success("✅ PDF uploaded and indexed!")

    # Chat interface
    st.subheader("💬 Ask a Question")
    user_input = st.text_input("Type your question here...")

    if st.button("Ask") and user_input and retriever:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        docs = retriever.get_relevant_documents(user_input)
        context = " ".join([d.page_content for d in docs[:3]])

        response = llm.invoke(f"Answer the question based on context:\n{context}\n\nQuestion: {user_input}")
        answer = response.content

        # ✅ Save in MongoDB
        save_to_db(user_input, answer)

        # Show answer
        st.markdown(f"**Answer:** {answer}")

    # ✅ Show MongoDB history
    st.subheader("📜 Recent History")
    history = load_history(limit=5)
    if history:
        for h in history:
            st.text(h)
            st.markdown("---")
    else:
        st.info("No history yet. Ask a question to start logging conversations.")


