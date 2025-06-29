from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_STORE_PATH = "chroma_db"

def create_or_load_vectorstore(docs=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Load if DB exists and docs are not passed
    if os.path.exists(VECTOR_STORE_PATH) and docs is None:
        print("🔁 Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )

    if docs is None:
        raise ValueError("No documents provided to create vector store.")

    print("🆕 Creating new Chroma vector store...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    db.persist()
    return db
