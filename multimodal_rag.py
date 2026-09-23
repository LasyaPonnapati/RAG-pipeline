import json
import os
from dotenv import load_dotenv

from openai import OpenAI
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import CompositeElement
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

MULTIMODAL_DB_DIR = "multimodal-rag-db"

def unstructured_partition_pdf(file_path: str):
    """Unstructured partition a PDF file into a list of documents."""
    print(f"Partitioning PDF file: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        # Detects layout (text, tables, figures) with a vision model. More accurate than "fast"; slower.
        strategy="hi_res",
        # Rebuilds tables as structured HTML/cells instead of a flat text dump.
        infer_table_structure=True,
        # Pulls figure/photo blocks out as Image elements (not tables or other types).
        extract_image_block_types=["Image"],
        # Stores each image's bytes on the element (base64 in metadata) so a multimodal LLM can see it later.
        extract_image_block_to_payload=True,
    )
    return elements

def unstructured_chunk_by_title(elements: list):
    """Chunk the elements by title."""
    chunks = chunk_by_title(
        elements=elements,
        # Maximum number of characters in a chunk
        max_characters=3000,
        # try to start a new chunk after n characters(avoid splitting in the middle of a sentence)
        new_after_n_chars=2400,
        # Combine text under n characters (to avoid small chunks)
        combine_text_under_n_chars=500,
    )
    return chunks

def get_content_data(chunk: CompositeElement):
    """Get the content data of a chunk."""
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"]
    }
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                content_data["types"].append("image")
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["images"].append(element.metadata.image_base64)

    return content_data

def create_ai_summary_for_chunk(text: str, tables: list, images: list):
    """Create an AI summary for a chunk."""
    SUMMARY_SYSTEM_PROMPT = (
        "You write a standalone summary of one document chunk for search and retrieval. "
        "The summary is stored on its own, so a reader who never sees the source must "
        "still understand it. Do not say \"this chunk\", \"the table above\", or \"the figure\".\n"
        "Cover:\n"
        "- The topic and the claim the passage makes, in the source's own terms\n"
        "- Facts worth retrieving: names, numbers, units, equations, and comparisons\n"
        "- Each table as sentences: column meaning, row relationships, and notable values. Do not copy raw HTML\n"
        "- Each image as a description: what it shows, labels or axes, and the point it supports\n"
        "Use only the text, tables, and images provided. Do not add outside knowledge. "
        "Write dense prose. Return only the summary."
    )
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",)
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Text: {text}\nTables: {tables}\nImages: {images}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error creating AI summary for chunk: {e}")
        return text

def summarize_chunks(chunks: list):
    langchain_docs = []
    """Summarize the chunks."""
    for chunk in chunks:
        content_data = get_content_data(chunk)
        if content_data["tables"] or content_data["images"]:
            print("creating AI summary for chunk with tables or images")
            enhanced_content = create_ai_summary_for_chunk(
                content_data["text"],
                content_data["tables"],
                content_data["images"]
            )
            print(f"AI summary: {enhanced_content[:200]}...")
        else:
            print("using raw text(chunk has no tables or images)")
            enhanced_content = content_data["text"]
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content":json.dumps({
                    "text": content_data["text"],
                    "tables": content_data["tables"],
                    "images": content_data["images"]
                })
            }
        )
        langchain_docs.append(doc)
    return langchain_docs

def store_docs(docs: list[Document], persist_directory: str = "multimodal-rag-db") -> Chroma:
    """Embed the summarized docs and save them in a separate Chroma folder."""
    print(f"Storing {len(docs)} documents in {persist_directory}...")

    os.makedirs(persist_directory, exist_ok=True)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

def main():
    """Main function."""
    print("Starting RAG pipeline...")

    # Partition the PDF file
    elements = unstructured_partition_pdf("multimodal-rag-docs/attention-is-all-you-need-Paper.pdf")

    # Chunk the elements by title - chunks are of type CompositeElement
    chunks = unstructured_chunk_by_title(elements)

    #summarize the chunks
    docs = summarize_chunks(chunks)
    print(docs[0].page_content)

    # Embed and persist in a Chroma database separate from db/chroma_db
    # store_docs(docs)

if __name__ == "__main__":
    main()
