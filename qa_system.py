from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')


loader = PyPDFLoader("instruction.pdf")
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
print(vector_store)