import os
from src.pdf_loader import load_pdfs
from src.chunker import chunk_documents, load_chunks
from src.embed_store import build_vector_store
from src.rag_agent import create_retriever_and_llm
from src.langgraph_agent import build_langgraph_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_DIR = "outputs/chroma_db"
CHUNKS_FILE = "outputs/chunks.json"

def main():
    # ---------- Step 1: Vector store ----------
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Vector store exists. Skipping PDF loading, chunking, and embedding")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    else:
        # Load PDFs and chunk
        if os.path.exists(CHUNKS_FILE):
            chunks = load_chunks(CHUNKS_FILE)
        else:
            docs = load_pdfs("pdfs/")
            chunks = chunk_documents(docs, save_path=CHUNKS_FILE)
        vectorstore = build_vector_store(chunks, persist_directory=VECTORSTORE_DIR)

    # ---------- Step 2: Retriever + LLM ----------
    retriever, llm = create_retriever_and_llm(vectorstore, k=10)

    # ---------- Step 3: LangGraph Agent ----------
    agent = build_langgraph_agent(retriever, llm, max_retries=2)

    # ---------- Step 4: Interactive queries ----------
    print("\n System ready. Type your questions below.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Enter your query: ").strip()
        if q.lower() in {"exit", "quit"}:
            print(" Exiting...")
            break

        state = {"question": q, "retrieved_docs": [], "answer": "", "attempts": 0}
        final_state = agent.invoke(state)

        print(f"\nQuery: {q}")
        ans = final_state.get("answer", "")
        rdocs = final_state.get("retrieved_docs", []) or []

        if rdocs:
            sources = [f"{d.metadata.get('source')} - page {d.metadata.get('page')}" for d in rdocs[:5]]
            ans += f"\n\n(Sources: {', '.join(sources)})"
        elif rdocs == []:
            ans += "\n\n Contextual data from PDFs insufficient to fully support this answer."

        print("Answer:", ans)
        print("Attempts:", final_state.get("attempts", 0))
        if final_state.get("failure_reason"):
            print("Note:", final_state["failure_reason"])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
