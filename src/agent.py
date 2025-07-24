import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from guardrails import Guard

# Load API key
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load LLM and chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load guardrails schema
guard = Guard.from_rail("guardrails.xml")  # Adjust if needed

# --- Moderation ---
def moderate_question(query: str) -> bool:
    from openai import OpenAI
    client = OpenAI()
    response = client.moderations.create(input=query)
    return response.results[0].flagged

# --- Main Agent ---
def run_agent(query: str):
    # Step 1: Moderation check
    if moderate_question(query):
        print("⚠️ Your query was flagged by moderation. Please rephrase.")
        return

    # Step 2: Run QA chain (use invoke instead of run)
    result = qa_chain.invoke({"query": query})
    llm_response = result["result"]  # this gives the string answer

    # Step 3: Validate with Guardrails
    try:
        guard.validate(
            queries={"query": query},
            llm_output=llm_response  # Pass just the string, not a dict
        )
    except Exception as e:
        print(f"⛔️ Guardrail triggered: {e}")
        return

    # Step 4: Output
    print("🤖 Answer:")
    print(llm_response)

# --- CLI ---
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question about the Big 4 Australian banks (or type 'exit'):\n> ")
        if user_query.lower() == "exit":
            break
        run_agent(user_query)
