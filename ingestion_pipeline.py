import os
import pypdf
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(path: str) -> list[Document]:
    print(f"Loading documents from {path}...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist")
    
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    if not documents:
        raise ValueError(f"No documents found in {path}")
    
    return documents

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    print(f"Splitting documents into chunks of {chunk_size} with {chunk_overlap} overlap...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError(f"No chunks found")
    
    return chunks

def create_vector_store(chunks: list[Document], persist_directory: str) -> Chroma:
    print(f"Creating vector store in {persist_directory}...")

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        api_key=os.getenv("GEMINI_API_KEY"),
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vector_store

def main():
    documents = load_documents("docs")

    chunks = split_documents(documents, 500, 0)

    vector_store = create_vector_store(chunks, "db/chroma_db")

if __name__ == "__main__":
    main()