import fitz

PDF_PATH = "data/sample.pdf"
CHUNK_SIZE = 300
OUTPUT_FILE = "chunks.txt"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def split_into_chunks(text, size):
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]

text = extract_text(PDF_PATH)
chunks = split_into_chunks(text, CHUNK_SIZE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"[CHUNK {i}]\n")
        f.write(chunk.strip() + "\n\n")

print(f"Exported {len(chunks)} chunks to chunks.txt")
