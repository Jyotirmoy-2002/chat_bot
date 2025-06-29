import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_FOLDER = "data"

def load_and_split_all_documents():
    all_docs = []
    print(f"📁 Checking folder: {DATA_FOLDER}")

    if not os.path.exists(DATA_FOLDER):
        print("❌ 'data/' folder does not exist.")
        return []

    files = os.listdir(DATA_FOLDER)
    if not files:
        print("⚠️ No files found in 'data/' folder.")
        return []

    for filename in files:
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"📄 Found file: {filename}")

        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                print(f"❌ Unsupported file format: {filename}")
                continue

            docs = loader.load()
            print(f"✅ Loaded {len(docs)} documents from {filename}")

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            print(f"🔪 Split into {len(chunks)} chunks.")
            all_docs.extend(chunks)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"📦 Total chunks loaded: {len(all_docs)}")
    return all_docs
