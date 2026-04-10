# from main import response
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
url = "https://en.wikipedia.org/wiki/Machine_learning"

template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarize the text."),
    ("human", "{data}"),
])

model = ChatMistralAI(model = "mistral-small-2603")
loader = WebBaseLoader(url)
data = loader.load()
prompt = template.format(data=data[0].page_content)
response = model.invoke(prompt)
print(response.content)