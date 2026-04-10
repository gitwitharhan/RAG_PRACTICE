from dotenv import load_dotenv
from langchain_mistralai import ChatmistralAI
load_dotenv()


model = ChatmistralAI(model = "mistral-small-2603")


response = model.invoke("Hello")

print(response.content) 