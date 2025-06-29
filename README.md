# chat_bot
This is a basic chat bot interface using chainlit that retrives data from uploaded pdf document, and answers relevant question.
this uses RAG, langchain framework, GROQ API key.
what does this do ?
-it reads the document.
-converts it into chunks, stored it in vector_db.
-calling LLM (llama-17b..), using GROQ API to call the model.
-The model takes the user query converts it into vector matches with specific vector index in chromadb(Retrives), then it generate the answer based on this.
-
