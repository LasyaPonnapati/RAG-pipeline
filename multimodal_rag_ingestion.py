import base64
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

GROQ_TEXT_MODEL = "openai/gpt-oss-20b"
GEMINI_VISION_MODEL = "gemini-3.6-flash"

SUMMARY_SYSTEM_PROMPT = (
    "You rewrite one document chunk as plain text in the same form as the surrounding paper. "
    "The result is stored beside chunks that were copied verbatim, so it must look like those chunks.\n"
    "Format:\n"
    "- Start with the section heading when the source has one, then continue in paragraphs\n"
    "- Write plain sentences. No markdown, no bold labels, no bullet lists, and no title like "
    "\"Encoder Structure\"\n"
    "- Do not say \"this passage\", \"this text\", \"the table\", or \"the figure\"\n"
    "Content:\n"
    "- Keep the source's terms, numbers, units, and equations\n"
    "- Turn each table into sentences in that same paragraph flow: what the columns mean and the notable values\n"
    "- Fold each image into the same paragraphs: name its parts, labels, and how they connect, "
    "in the order they appear\n"
    "Use only the text, tables, and images provided. Return only that text."
)


def _image_bytes(image_b64: str) -> tuple[bytes, str]:
    """Decode a raw or data-URL base64 image for Gemini."""
    raw = image_b64.strip()
    if raw.startswith("data:"):
        header, raw = raw.split(",", 1)
        mime = header.split(";")[0].removeprefix("data:") or "image/jpeg"
    elif raw.startswith("iVBOR"):
        mime = "image/png"
    elif raw.startswith("/9j/"):
        mime = "image/jpeg"
    elif raw.startswith("R0lGOD"):
        mime = "image/gif"
    elif raw.startswith("UklGR"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return base64.b64decode(raw), mime


def _summarize_with_gemini(prompt: str, images: list) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment (.env)")

    parts: list = [prompt]
    for image in images:
        if not image:
            continue
        data, mime = _image_bytes(image)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
        ),
    )
    return response.text or ""


def _summarize_with_groq(prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def create_ai_summary_for_chunk(text: str, tables: list, images: list):
    """Summarize a chunk. Images use Gemini; tables stay on the Groq text model."""
    prompt = f"Text: {text}\nTables: {tables}"
    image_inputs = [image for image in images if image]
    try:
        if image_inputs:
            print(f"summarizing with {GEMINI_VISION_MODEL}")
            return _summarize_with_gemini(prompt, image_inputs)
        print(f"summarizing with {GROQ_TEXT_MODEL}")
        return _summarize_with_groq(prompt)
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
    store_docs(docs)

if __name__ == "__main__":
    main()
