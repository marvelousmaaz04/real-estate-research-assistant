from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import shutil
import os

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    """Initialize LLM and Vector Store (fresh per run)."""
    global llm, vector_store

    # Initialize LLM
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=500)

    # Clear old persisted vectorstore
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)

    ef = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )

    # In-memory store (no persistence)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=None
    )


def process_urls(urls):
    """Fetch, split, embed, and store documents."""
    yield "Initializing components..."
    initialize_components()

    yield "Loading HTML data...✅"
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    yield "Converting HTML to text...✅"
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    yield "Splitting text into chunks...✅"
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    docs_split = splitter.split_documents(docs_transformed)

    for d in docs_split:
        d.metadata["source"] = d.metadata.get("source") or "Unknown"

    yield "Adding documents to vector store...✅"
    uuids = [str(uuid4()) for _ in docs_split]
    vector_store.add_documents(docs_split, ids=uuids)

    yield "✅ Processing complete! You can now ask questions."


def generate_answer(query):
    """Generate answers based on the embedded documents."""
    if not vector_store:
        raise RuntimeError("Vector database is not initialized.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result["answer"], sources
