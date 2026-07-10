from langchain_openai import OpenAI
from dotenv import load_dotenv

# env variable
load_dotenv()

# model name
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# invoke hit the model and give the prompt to model 
result = llm.invoke("What is the capital of India")

print(result) # donot have openai api key
