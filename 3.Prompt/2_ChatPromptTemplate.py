from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

result = chat_template.invoke({
    'domain': 'programming',
    'topic': 'HashMap'
})

print(result)
