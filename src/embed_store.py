from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vector_store(chunks, persist_directory="outputs/chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Debug: print chunk stats
    for i, text in enumerate(texts[:5]):  
        print(f"Chunk {i}: length={len(text)} chars")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    vectorstore.persist()

    print(f"Vector store built with {len(chunks)} chunks at {persist_directory}")
    return vectorstore
