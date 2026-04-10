from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


data = TextLoader("document_loaders/notes.txt").load()
ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
    ("user", "{context}")
])


model = ChatMistralAI(model = "mistral-small-2603")


response = model.invoke("Hello")

print(response.content) 