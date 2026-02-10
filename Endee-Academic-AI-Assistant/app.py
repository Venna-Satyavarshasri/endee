import streamlit as st
import fitz
import numpy as np
import faiss
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 350          # Bigger chunks = better structure
TOP_K = 8                # Pull more context
OCR_DPI = 300

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Endee Academic AI Assistant",
    layout="centered"
)

# ================= UI STYLE =================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { max-width: 900px; padding-top: 2rem; }

.card {
    background-color: #161b22;
    padding: 1.6rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    border: 1px solid #30363d;
}

.answer-box {
    background-color: #0d1117;
    padding: 1.4rem;
    border-radius: 12px;
    border-left: 5px solid #2f81f7;
    white-space: pre-wrap;
    line-height: 1.6;
}

.status { color: #9da7b3; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="card">
<h1>📘 Endee Academic AI Assistant</h1>
<p class="status">
Upload any academic PDF and ask questions.<br>
Answers are extracted directly from the document — clean, ordered, exam-ready.
</p>
</div>
""", unsafe_allow_html=True)

# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= OCR =================
def ocr_pdf(doc):
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        full_text += pytesseract.image_to_string(img, lang="eng") + "\n"
    return full_text

# ================= CHUNKING =================
def chunk_text(text):
    return [
        text[i:i + CHUNK_SIZE].strip()
        for i in range(0, len(text), CHUNK_SIZE)
        if len(text[i:i + CHUNK_SIZE].strip()) > 40
    ]

# ================= ORDER RESTORATION =================
def restore_document_order(chunks, all_chunks):
    index_map = {c: i for i, c in enumerate(all_chunks)}
    chunks.sort(key=lambda c: index_map.get(c, 10**9))
    return chunks

# ================= CLEAN FORMAT =================
def clean_and_format_answer(text):
    lines, seen = [], set()

    for line in text.split("\n"):
        line = line.strip()

        if len(line) < 4:
            continue

        # normalize broken numbering
        if line[0].isdigit() and "." in line[:3]:
            line = "\n" + line

        # remove duplicates
        if line not in seen:
            seen.add(line)
            lines.append(line)

    return "\n".join(lines).strip()

# ================= SESSION =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# ================= PDF UPLOAD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Upload PDF")

uploaded_pdf = st.file_uploader(
    "Scanned PDFs, notes, question banks supported",
    type=["pdf"]
)

if uploaded_pdf:
    with st.spinner("🔍 Extracting text from PDF..."):
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text = ocr_pdf(doc)

    chunks = chunk_text(text)

    if not chunks:
        st.error("No readable content detected.")
    else:
        embeddings = model.encode(chunks, show_progress_bar=False)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))

        st.session_state.chunks = chunks
        st.session_state.index = index

        st.success(f"PDF processed • {len(chunks)} sections indexed")

st.markdown("</div>", unsafe_allow_html=True)

# ================= QUESTION =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Ask a Question")

question = st.text_input(
    "Example: Explain ultrasonic machining",
    placeholder="Type your academic question..."
)
st.markdown("</div>", unsafe_allow_html=True)

# ================= ANSWER =================
if question:
    if st.session_state.index is None:
        st.warning("Upload a PDF first.")
    else:
        with st.spinner("Extracting relevant sections..."):
            q_vec = model.encode([question])
            _, indices = st.session_state.index.search(
                np.array(q_vec).astype("float32"),
                TOP_K
            )

            retrieved = [st.session_state.chunks[i] for i in indices[0]]

            ordered = restore_document_order(
                retrieved,
                st.session_state.chunks
            )

            final_answer = clean_and_format_answer(
                "\n\n".join(ordered)
            )

        st.markdown("""
        <div class="card">
        <h3>Answer</h3>
        <div class="answer-box">
        """ + final_answer + """
        </div>
        </div>
        """, unsafe_allow_html=True)
