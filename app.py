import chainlit as cl
from load_documents import load_and_split_all_documents
from vector_store import create_or_load_vectorstore
from qa_chain import build_qa_chain

@cl.on_chat_start
async def start():
    try:
        # Load Chroma DB from disk
        vectorstore = create_or_load_vectorstore()
    except Exception as e:
        print(f"⚠️ Could not load existing vectorstore: {e}")
        # If DB doesn't exist, load documents and build it
        docs = load_and_split_all_documents()
        vectorstore = create_or_load_vectorstore(docs)

    qa_chain = build_qa_chain(vectorstore)
    cl.user_session.set("qa", qa_chain)

    await cl.Message(content="📚 Assistant is ready! Ask me anything from the document!").send()

@cl.on_message
async def main(msg: cl.Message):
    qa = cl.user_session.get("qa")
    answer = qa.run(msg.content)
    await cl.Message(content=answer).send()
