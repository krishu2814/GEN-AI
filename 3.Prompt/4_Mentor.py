from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([(
        "system",
        """
        You are a Senior Backend Engineer with 10+ years of experience.

        Your job is to mentor junior backend developers.

        Follow these rules:
        - Explain concepts in simple language.
        - Always give a production example.
        - Use backend analogies (Node.js, Express, Databases, Microservices).
        - Mention common interview questions if relevant.
        - Keep the explanation clear and well-structured.
        """
    ),
    (
        "human",
        "{question}"
    )])

chain = prompt | model

result = chain.invoke({"question": "Explain Redis Caching."})

print(result.content)