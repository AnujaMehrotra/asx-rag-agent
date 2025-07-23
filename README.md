# 🧠 ASX RAG Agent – GenAI Project on Top 4 Australian Banks

This project demonstrates a Retrieval-Augmented Generation (RAG) agent built using public annual reports from the top four Australian banks: **CBA**, **Westpac**, **NAB**, and **ANZ**.

It answers natural language questions by retrieving context from embedded annual report documents and using GPT-4 (or GPT-3.5) to generate grounded responses.

---

## 🔍 What This Project Demonstrates

- ✅ RAG (Retrieval-Augmented Generation) pipeline using LangChain
- ✅ PDF parsing and text chunking
- ✅ OpenAI Embedding + FAISS vector store
- ✅ Guardrails via custom prompts and moderation
- ✅ CLI-based interaction with `.sys`-style execution
- ✅ Built with future readiness for agentic workflows (LangGraph-ready)

---

## 🏦 Data Sources

- 📄 2023 Annual Reports for CBA, Westpac, NAB, ANZ
- 📚 Wikipedia summaries for each bank
- All documents saved in the `data/` folder

---

## 🛠️ Tech Stack

| Component | Tool |
|----------|------|
| Embedding Model | OpenAI (`text-embedding-ada-002`) |
| Vector Store | FAISS |
| LLM | OpenAI GPT (via LangChain) |
| PDF Parsing | `pdfplumber` |
| Prompt Engineering | Custom templates with guardrails |
| Framework | LangChain |
| Moderation | OpenAI Moderation API |
| CLI Interface | Python `.sys` entry point |

---

## 📁 Project Structure

```bash
.
├── data/                  # PDFs and extracted text
├── src/
│   ├── ingest.py          # Scrapes Wikipedia summaries
│   ├── parse_pdf.py       # Extracts text from PDFs
│   ├── embed.py           # Embeds text into FAISS
│   ├── agent.py           # Runs RAG question-answering
│   └── main.py            # CLI entry point
├── .env                   # API keys (excluded from Git)
├── requirements.txt
├── .gitignore
├── README.md
└── run_agent.sys          # Example CLI usage
git add