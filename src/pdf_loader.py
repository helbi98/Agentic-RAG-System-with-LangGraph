import pdfplumber
import os
import hashlib
from PIL import Image
import pytesseract

# Optional: if tesseract is not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def normalize_text(text: str) -> str:
    """whitespace cleanup for comparison and storage."""
    return " ".join(text.split())

def hash_text(text: str) -> str:
    """Hash text for duplicate detection."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_pdfs(pdf_dir="pdfs/"):
    documents = []
    debug_lines = []

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, fname)
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                debug_lines.append(f"\n--- File: {fname} | Page {page_num} ---\n")
                seen_hashes = set()
                has_plain_text = False

                # --- Extract plain text ---
                text = page.extract_text() or ""
                if text.strip():
                    clean = normalize_text(text)
                    h = hash_text(clean)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        has_plain_text = True
                        debug_lines.append("[PLAIN TEXT]")
                        debug_lines.append(clean)
                        documents.append({
                            "content": clean,
                            "metadata": {"source": fname, "page": page_num, "type": "paragraph"}
                        })
                else:
                    debug_lines.append("[NO TEXT EXTRACTED BY PDFPLUMBER]")

                # --- Extract tables only if no plain text found---
                if not has_plain_text:
                    tables = page.extract_tables()
                    for table in tables:
                        rows = [" | ".join([cell if cell else "" for cell in row]) for row in table]
                        for row in rows:
                            clean = normalize_text(row)
                            if clean:
                                h = hash_text(clean)
                                if h not in seen_hashes:
                                    seen_hashes.add(h)
                                    debug_lines.append(f"[TABLE ROW] {clean}")
                                    documents.append({
                                        "content": clean,
                                        "metadata": {"source": fname, "page": page_num, "type": "table"}
                                    })

                # --- OCR only if no plain text found---
                if not has_plain_text:
                    for img_index, img in enumerate(page.images):
                        im = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(im, lang='eng')
                        clean = normalize_text(ocr_text)
                        if clean:
                            h = hash_text(clean)
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                debug_lines.append("[OCR TEXT]")
                                debug_lines.append(clean)
                                documents.append({
                                    "content": clean,
                                    "metadata": {"source": fname, "page": page_num, "type": "ocr_table"}
                                })

    # Debug file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/raw_text_debug.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines))

    print("Raw text written to outputs/raw_text_debug.txt")
    return documents

