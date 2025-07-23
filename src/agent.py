import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from openai import OpenAI as OpenAIClient
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's in your .env file.")

# Setup OpenAI client
openai_client = OpenAIClient()

VECTOR_DB_PATH = "data/faiss_index"

# Guardrail prompt template
template = """
You are a helpful assistant answering questions ONLY based on the provided bank annual report documents.
If the question is unrelated to the documents, respond: "Sorry, I can only answer questions about these bank reports."
Avoid speculation or generating information not in the documents.

Question: {question}
=========
{context}
=========
Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def moderate_question(question):
    response = openai_client.moderations.create(input=question)
    flagged = response["results"][0]["flagged"]
    return flagged

def main():
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM with guardrails: deterministic and limited response length
    llm = OpenAI(temperature=0, max_tokens=500)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )

    print("Ready for questions! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if moderate_question(query):
            print("Sorry, your question violates usage policies. Try a different question.")
            continue

        answer = qa_chain.run(query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
