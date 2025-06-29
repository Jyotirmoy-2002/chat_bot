from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # ✅ Ensure this is installed
import yaml

def load_prompt_template(path="prompt.yaml"):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return PromptTemplate(
        input_variables=["context", "question"],
        template=data["template"]
    )

def build_qa_chain(vectorstore):
    # 🔐 Load your Groq API key from environment variable
    import os
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )

    prompt = load_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain
