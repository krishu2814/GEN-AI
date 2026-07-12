from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate # for string prompts
from langchain_core.prompts import ChatPromptTemplate # for multiple messages
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a backend engineering mentor."
        ),
        (
            "human",
            "Explain {topic} in simple words."
        )
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "HashMap"})

print(type(result))
print(result)
