from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant.
    use only the provided text to answer the question.
    if the ans is not present in the context ,
    say "I could not find the answer"
    """),
    ("human","""
    Context: {context}
    Question: {question}
    """)
])
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory="./.chroma_db", embedding_function=embedding)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5,"lambda_mult": 0.5}
)

model = ChatMistralAI(model = "mistral-small-2603")

print("Rag is Created")

print("Press 0 to exit")


while True:
    query = input("Ask a question: ")
    if query == "0":
        break
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = model.invoke(final_prompt)
    print(f"\nAnswer: {response.content}")  

    


  