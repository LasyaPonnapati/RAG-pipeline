import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

DOCS_PATH = "docs"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 0
GROQ_MODEL = "openai/gpt-oss-20b"

_groq_client: OpenAI | None = None

AGENTIC_SYSTEM_PROMPT = (
    "You split documents into chunks that belong together in meaning. "
    "Return the exact text that was given to you, and insert the marker <<<split>>> "
    "only at the boundaries between those chunks. "
    "Do not add commentary, markdown, or extra text. "
    "Do not rewrite, summarize, or omit content. "
    "Keep original wording, whitespace, and order. "
    "Do not put <<<split>>> at the very start or very end of the text."
)


def _get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


def load_documents(path: str) -> list[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist")

    documents = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader).load()
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents


def character_text_split(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Splits only on one separator (\\n\\n), then packs pieces until chunk_size.
    Overlap repeats the previous chunk's tail; a piece bigger than chunk_size may stay oversized."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def recursive_character_text_split(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Tries separators in order: paragraphs, lines, words, then characters.
    Drops to a finer separator only when a piece still exceeds chunk_size."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def semantic_text_split(documents: list[Document]) -> list[Document]:
    """Splits at topic shifts: embed sentences, then cut where neighboring similarity drops.
    Ignores chunk_size; needs an embedding API call and can produce uneven chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
    )
    return splitter.split_documents(documents)


def agentic_text_split(documents: list[Document]) -> list[Document]:
    """Asks an LLM to cut the text where meaning shifts and mark those cuts with <<<split>>>."""
    client = _get_groq_client()
    chunks: list[Document] = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            reasoning_effort="low",
            max_completion_tokens=8192,
            messages=[
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user", "content": doc.page_content},
            ],
        )
        message = response.choices[0].message
        marked = (message.content or "").strip()
        if not marked:
            print(
                "Agentic split: Groq returned empty content "
                f"(finish_reason={response.choices[0].finish_reason}); using original text."
            )
            chunks.append(doc)
            continue
        parts = [part.strip() for part in marked.split("<<<split>>>") if part.strip()]
        for part in parts:
            chunks.append(Document(page_content=part, metadata=dict(doc.metadata)))
    return chunks
    

def _print_chunks(title: str, chunks: list[Document], extra: str = "") -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    details = extra or f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}"
    print(f"{details}, n_chunks={len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5], start=1):
        print(f"--- chunk {i} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content)
        print()


def main() -> None:
    documents = load_documents(DOCS_PATH)
    char_chunks = character_text_split(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    recursive_chunks = recursive_character_text_split(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    semantic_chunks = semantic_text_split(documents)
    agentic_chunks = agentic_text_split(documents)

    _print_chunks("1) Character text splitting", char_chunks)
    _print_chunks("2) Recursive character text splitting", recursive_chunks)
    _print_chunks(
        "3) Semantic text splitting",
        semantic_chunks,
        extra="breakpoint_threshold_type=percentile",
    )
    _print_chunks(
        "4) Agentic text splitting",
        agentic_chunks,
        extra=f"model={GROQ_MODEL}",
    )


if __name__ == "__main__":
    main()
