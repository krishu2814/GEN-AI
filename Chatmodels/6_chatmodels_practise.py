from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

shopping_prompt = PromptTemplate.from_template(
"""
You are an expert shopping assistant.
Recommend a {category}
under ₹{budget}
for a {skill_level} programmer.
For each recommendation provide:

1. Laptop Name
2. Price
3. Specifications
4. Why it is suitable
5. Pros
6. Cons
"""
)

# to fill placeholders in the prompt template, we use invoke method of PromptTemplate class
final_prompt = shopping_prompt.invoke(
    {
        "category": "Coding Laptop",
        "budget": "70000",
        "skill_level": "Intermediate"
    }
)

print("Generated Prompt:\n")
print(final_prompt.text)

print("-" * 100)

response = model.invoke(final_prompt)

print(response.content)
