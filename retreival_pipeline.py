import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

def retrieve(db: Chroma, query: str) -> list[Document]:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return docs

def main():
    query = input("Enter a question: ")

    # Same embedding model used during ingestion — required to query existing DB
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Opens the EXISTING db (does not create a new one)
    db = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
    )

    docs = retrieve(db, query)
    for doc in docs:
        print(doc.page_content)
        print("-" * 100)

if __name__ == "__main__":
    main()
