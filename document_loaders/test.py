from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader


data = TextLoader("document_loaders/notes.txt").load()


print(data)