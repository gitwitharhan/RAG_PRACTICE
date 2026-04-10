from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


data = PyPDFLoader("document_loaders/ml.pdf").load()
template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarize the text."),
    ("human", "{data}"),
])

prompt = template.format(data=data[0].page_content) 


model = ChatMistralAI(model = "mistral-small-2603")

 
response = model.invoke(prompt )

print(response.content) 