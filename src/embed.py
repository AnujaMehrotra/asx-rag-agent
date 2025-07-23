import os
import glob
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

DATA_DIR = "data"
VECTOR_DB_PATH = "data/faiss_index"

def load_text_files():
    docs = []
    for filepath in glob.glob(os.path.join(DATA_DIR, "*_annual_2023.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            name = os.path.basename(filepath).split("_")[0].upper()
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

def embed_documents(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Saved vector store to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    raw_docs = load_text_files()
    print(f"Loaded {len(raw_docs)} annual report documents.")

    chunks = chunk_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    embed_documents(chunks)
