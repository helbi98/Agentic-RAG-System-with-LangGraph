import os
from langchain_groq import ChatGroq

def create_retriever_and_llm(vectorstore, k=8):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
        temperature=0,
    )
    return retriever, llm
