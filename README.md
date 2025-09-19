# MedicalGuides – Agentic RAG for Clinical Guidelines

This is an AI-powered question-answering system designed to provide grounded, cited answers based on clinical guideline PDFs.  
It uses an **Agentic Retrieval-Augmented Generation (RAG) pipeline** with **LangGraph** for orchestration, **SentenceTransformer embeddings** stored in **ChromaDB** for semantic search, and **Groq API-powered LLMs** for intelligent query routing, self-correction, and answer generation.

---

## Features
- **Agentic RAG pipeline** with multi-step control flow using LangGraph
- **Sentence-aware chunking** with character-based overlap for better context preservation
- **Semantic search** using SentenceTransformer embeddings stored in persistent ChromaDB
- **Query routing** – LLM decides whether a query needs document lookup or can be answered directly
- **Self-correcting query rewriting** – improves recall if no relevant context is found
- **Context-aware answer generation** with inline PDF source citations

---

## Architecture

### 1. Data Ingestion
- Loads and parses PDF guidelines.
- Extracts text and applies preprocessing (cleaning, normalization).
- Splits content into overlapping, sentence-aware chunks for optimal retrieval.

### 2. Vector Store
- Embeds each chunk using **SentenceTransformers**.
- Stores embeddings in a **ChromaDB persistent collection** for fast similarity search.
- Metadata (source filename, page number, etc) saved for later citation.

### 3. Retrieval
- Incoming query is embedded with the same model.
- Top-k most relevant chunks retrieved via cosine similarity from ChromaDB.

### 4. Agent Orchestration
- **Router Node:** LLM classifies query → `LOOKUP` (retrieve) or `DIRECT` (answer from general knowledge).
- **Retriever Node:** fetches chunks if lookup is required.
- **LLM Answer Node:** synthesizes a concise answer, citing context when used.
- **Self-Correction Node:** LLM rewrites the query (e.g., synonyms, German/English variants) and retries retrieval if needed.

### 5. Answer Generation
- Groq API LLM composes the final answer.
- Includes citations in the format `[filename - page]`.
- Falls back to general knowledge if no relevant context is available.

---

### Install Requirements

pip install -r requirements.txt


### Set up Environment Variables

setx GROQ_API_KEY "your_api_key_here" 


### Run the Pipeline

python -m src.run_pipeline