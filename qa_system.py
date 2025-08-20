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

# import date
from datetime import datetime, timezone
from pymongo import MongoClient


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')


mongo_uri = os.getenv("MONGO_URI")
# print("****************************************",mongo_uri)
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
chat_collection = db["chat_history"]

def save_to_db(question, answer):
    chat_collection.insert_one({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now(timezone.utc)
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


loader = PyPDFLoader("instruction2.pdf")
docs = loader.load()
# print(len(docs))

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = text_spliter.split_documents(docs)
# print(len(chunks))

embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
# print(vector_store)

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
    
    chat_history = load_history(limit=5)
    history_text = "\n".join(chat_history)
    
    context = history_text + "\n\n" + "\n\n".join(doc.page_content for doc in documents)
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

while True:
    user_input = input("You:")
    if user_input.lower() in ["exit", "bye"]:
        print("Come back Soon :)")
        break
    else:
        response= work_flow.invoke({'question': user_input})
        print(response['response'])



