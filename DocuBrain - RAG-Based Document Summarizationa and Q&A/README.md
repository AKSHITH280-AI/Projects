# PDF RAG Chatbot

This project is a Streamlit-based PDF chatbot built with the stack you requested:

- LangChain for the RAG pipeline
- GPT-4 as the language model
- ChromaDB for vector storage
- PyPDF2 and pdfplumber for PDF parsing
- Streamlit for the interface

It lets a user upload supported documents, ask grounded questions, and request a summary. The app is tuned around three practical retrieval controls in the UI:

- Top-k relevance
- Retrieval precision
- Quality

## How It Works

1. Uploaded files are routed by extension to the correct loader.
2. PDFs are parsed page by page using both `PyPDF2` and `pdfplumber`.
3. DOCX files are parsed with `python-docx`.
4. Excel files are parsed sheet by sheet with `pandas`.
5. TXT and CSV files are converted into plain text.
6. Extracted text is split into chunks with LangChain.
7. Chunks are embedded and stored in a local ChromaDB collection.
8. For each question, the app retrieves the top-k most relevant chunks.
9. Low-relevance results are filtered out to improve precision and reduce hallucination.
10. GPT-4 answers using only the retrieved document context.
11. The UI shows citations plus retrieval-quality signals.

## Supported Formats

- PDF
- DOCX
- XLSX
- TXT
- CSV

## Setup

1. Create a virtual environment if you want:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from `.env.example`:

   ```bash
   cp .env.example .env
   ```

4. Add your OpenAI API key to `.env`.

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Environment Variables

- `OPENAI_API_KEY`: required
- `LLM_PROVIDER`: use `openai` or `ollama`
- `OPENAI_MODEL`: defaults to `gpt-4`
- `OLLAMA_MODEL`: defaults to `llama3.1`
- `EMBEDDING_PROVIDER`: use `local` to avoid OpenAI embedding quota, or `openai` to use OpenAI embeddings
- `OPENAI_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`
- `LOCAL_EMBEDDING_MODEL`: defaults to `sentence-transformers/all-MiniLM-L6-v2`

## LLM Providers

Use OpenAI for best answer quality:

```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4
```

Use Ollama for local, quota-free answering:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

Use Groq for fast hosted inference:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Use MiniMax:

```env
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your_minimax_api_key_here
MINIMAX_MODEL=MiniMax-M2.5
MINIMAX_BASE_URL=https://api.minimax.io/v1
```

Use NVIDIA:

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

For Ollama mode, install and run Ollama separately:

```bash
brew install ollama
ollama pull llama3.1
ollama serve
```

## Embedding Providers

The app supports two embedding modes:

- `EMBEDDING_PROVIDER=local`: uses a local SentenceTransformers model for document indexing. This avoids OpenAI embedding quota errors during upload.
- `EMBEDDING_PROVIDER=openai`: uses OpenAI embeddings for document indexing.

The recommended project/demo setting is:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The first local run may download the model once.

## Hallucination Control

- The answer prompt explicitly forbids outside knowledge.
- Only retrieved PDF chunks are passed to the model for Q&A.
- A minimum relevance threshold is applied before answering.
- If support is weak, the assistant returns a refusal instead of guessing.
- Page citations are shown for verification.

## Scanned PDFs and OCR

- The extractor first tries normal text extraction with `PyPDF2` and `pdfplumber`.
- If the extracted content looks too weak, the pipeline can attempt OCR recovery for scanned or image-based PDFs.
- OCR is optional and uses `pytesseract` when available.
- For OCR recovery to work fully on your machine, the Tesseract OCR binary must also be installed.

## Retrieval Quality Notes

- `Top-k relevance` controls how many chunks are considered.
- `Retrieval precision` is shown as the fraction of top-k chunks that cleared the relevance threshold.
- `Quality` is a simple label derived from average relevance and precision so you can quickly judge whether the retrieved context looks strong enough.

## Actual Precision Evaluation

If you need true retrieval precision for your project report, use the offline evaluation script instead of the live UI heuristic.

1. Put the PDFs you want to evaluate inside an `evaluation_pdfs/` folder.
2. Add labeled questions to [evaluation_dataset.json](/Users/deekshithamandhadi/Documents/New%20project/evaluation_dataset.json#L1) with the relevant `chunk_id` values.
3. Run:

   ```bash
   python evaluate.py --pdf-dir evaluation_pdfs --dataset evaluation_dataset.json --top-k 5
   ```

The script computes actual:

- `Precision@k`
- `Recall@k`
- `F1@k`

### How chunk ids work

Each chunk now gets a stable id in this format:

```text
source.pdf_p<page>_c<chunk_number>
```

Example:

```text
bert.pdf_p6_c1
```

Use those ids in the evaluation dataset as ground-truth relevant chunks for each question.

### How to list chunk ids

To inspect chunk ids before labeling your dataset, run:

```bash
python list_chunks.py --pdf-dir evaluation_pdfs
```

This prints, for every chunk:

- `chunk_id`
- source PDF
- page number
- short excerpt
