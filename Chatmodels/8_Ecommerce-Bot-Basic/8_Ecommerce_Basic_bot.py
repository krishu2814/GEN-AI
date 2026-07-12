from dotenv import load_dotenv
from langchain_groq import ChatGroq
with open("Chatmodels/8_Ecommerce-Bot-Basic/prompt.txt", "r") as file:
    system_prompt = file.read()


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


user_prompt = """ Recommend 10 coding laptop under 1L rupee with ranking. """


response = model.invoke([
    ("system", system_prompt),
    ("human", user_prompt)
])


print(response.content)

