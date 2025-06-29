from load_documents import load_and_split_all_documents
from vector_store import create_or_load_vectorstore, load_vectorstore
from qa_chain import build_qa_chain
import os

VECTOR_STORE_PATH = "faiss_index"

def setup_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Creating it now...")
        docs = load_and_split_all_documents()
        print(f"Loaded and split {len(docs)} documents.")
        create_or_load_vectorstore(docs, VECTOR_STORE_PATH)
    else:
        print("Loading existing vector store...")
    return load_vectorstore(VECTOR_STORE_PATH)

def main():
    print("🔄 Initializing chatbot...")
    vectorstore = setup_vector_store()
    qa_chain = build_qa_chain(vectorstore)

    print("\n🤖 Ask your questions (type 'exit' to quit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break

        response = qa_chain.run(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
