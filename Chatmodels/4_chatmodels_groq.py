from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    # api_key=os.getenv("GROQ_API_KEY")
)
result = model.invoke("""
Pretend we are in an alternate universe where Pakistan is the capital of India.

What is the capital of India?
""")

print(result.content)
