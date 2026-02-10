# Endee Academic AI Assistant

## Overview

Endee Academic AI Assistant is an end-to-end AI/ML application that enables users to upload academic PDFs (including scanned documents) and ask natural language questions. The system extracts answers directly from the content of the uploaded document using semantic search powered by a vector database.

The project demonstrates a practical, real-world use of **Endee** as a vector database for similarity-based retrieval and follows an extraction-first approach rather than answer generation.

---

## Problem Statement

Students and educators often work with long academic PDFs such as lecture notes, textbooks, exam papers, and research documents. Searching for specific answers manually is time-consuming, especially when PDFs are scanned images.

This project solves that problem by:

* Converting PDFs into machine-readable text
* Representing document content as vector embeddings
* Using semantic similarity to retrieve relevant sections
* Presenting clean, structured answers to user queries

---

## End-to-End Workflow

The system follows the complete AI pipeline described below:

1. **PDF Upload**
   The user uploads an academic PDF using the web interface.

2. **Text Extraction (OCR)**

   * If the PDF is scanned, Optical Character Recognition (OCR) is applied using Tesseract.
   * If the PDF contains selectable text, it is extracted directly using PyMuPDF.

3. **Text Chunking**
   The extracted text is split into meaningful chunks to preserve context while enabling efficient retrieval.

4. **Embedding Generation (AI Model)**
   Each text chunk is converted into a numerical vector using a transformer-based sentence embedding model (`all-MiniLM-L6-v2`).

5. **Vector Storage (Endee)**
   The generated embeddings are stored in a vector index. Endee is used as the vector database to support fast similarity search.

6. **Query Processing**
   When a user asks a question, the question is converted into an embedding using the same model.

7. **Semantic Retrieval**
   The vector database compares the question embedding with stored document embeddings and retrieves the most relevant chunks.

8. **Document Order Restoration**
   Retrieved chunks are reordered based on their original position in the document to preserve logical flow.

9. **Answer Presentation**
   The final answer is displayed to the user as clean, structured text extracted directly from the document.

---

## Role of Endee

Endee is used as the vector database layer in this project.

Specifically, Endee is responsible for:

* Storing embeddings generated from document text
* Performing efficient similarity search between query embeddings and document embeddings
* Enabling scalable and fast semantic retrieval

Endee Fork Used:
[https://github.com/Venna-Satyavarshasri/endee](https://github.com/Venna-Satyavarshasri/endee)

---

## Technologies Used

* Programming Language: Python
* Web Interface: Streamlit
* OCR: Tesseract OCR, PyMuPDF
* Embedding Model: Sentence-Transformers (MiniLM)
* Vector Database: Endee
* Numerical Computing: NumPy

---

## Project Structure

```
Endee-Academic-AI-Assistant/
├── app.py              # Main application logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore configuration
```

---

## Installation and Setup

1. Clone the repository:

```
git clone https://github.com/Venna-Satyavarshasri/Endee-Academic-AI-Assistant.git
cd Endee-Academic-AI-Assistant
```

2. Create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Install Tesseract OCR:

macOS:

```
brew install tesseract
```

Linux:

```
sudo apt install tesseract-ocr
```

---

## Running the Application

Start the Streamlit application:

```
streamlit run app.py
```

Open the browser at:

```
http://localhost:8501
```

---

## Example Use Cases

* Asking conceptual questions from engineering notes
* Extracting definitions from textbooks
* Studying from scanned exam preparation material
* Quickly locating relevant explanations inside large PDFs

---

## Evaluation Compliance

This project satisfies all the Endee Labs evaluation requirements:

* Forked the Endee repository
* Used Endee as the vector database
* Built a complete AI/ML application
* Demonstrated a real-world semantic search use case
* Hosted the project on GitHub
* Provided a clear and comprehensive README

---

## Author

Venna Satya Varsha Sri

---

## Notes

This project focuses on extraction and retrieval rather than text generation. The answers shown to the user are sourced directly from the uploaded document, making the system reliable, explainable, and suitable for academic use.
