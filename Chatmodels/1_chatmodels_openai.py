from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# temp -> 0 to 2 -> creative response
# max_completion_tokens -> tokens in output
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content) # donot have openai api key 
