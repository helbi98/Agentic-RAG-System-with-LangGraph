from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import os
import json

nltk.download('punkt', quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

def smart_sentence_chunks(text, max_size=1500, overlap=3):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1 
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            overlap_sentences = current_chunk[-overlap:] if overlap > 0 and len(current_chunk) >= overlap else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

def chunk_documents(documents, chunk_size=1500, chunk_overlap=3, save_path="outputs/chunks.json"):
    """
    Chunk documents with:
    - Deduplication
    - Sentence-aware splitting for large paragraphs or table rows
    Saves the resulting chunks as JSON for reuse.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = []
    seen_texts = set()

    for doc in documents:
        content = doc["content"]
        doc_type = doc["metadata"].get("type", "paragraph")

        if len(content) > chunk_size:
            splits = smart_sentence_chunks(content, max_size=chunk_size, overlap=chunk_overlap)
        else:
            splits = [content]

        refined_splits = []
        for split in splits:
            if len(split) > chunk_size:
                refined_splits.extend(text_splitter.split_text(split))
            else:
                refined_splits.append(split)

        for i, split in enumerate(refined_splits):
            clean = split.strip()
            if clean and clean not in seen_texts:
                seen_texts.add(clean)
                chunks.append({
                    "content": clean,
                    "metadata": {**doc["metadata"], "chunk": i}
                })

    # Save chunks to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(chunks)} chunks saved to {save_path}")
    return chunks

def load_chunks(save_path="outputs/chunks.json"):
    """Load previously saved chunks from JSON"""
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {save_path}")
        return chunks
    else:
        print(f"No chunk file found at {save_path}")
        return []
