import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

VECTOR_DB_PATH = "data/faiss_index"

def main():
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = OpenAI(temperature=0, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("Ready for questions! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
