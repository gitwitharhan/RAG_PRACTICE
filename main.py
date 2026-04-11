from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()



template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarize the text."),
    ("human", "{data}"),
])




model = ChatMistralAI(model = "mistral-small-2603")

 


 