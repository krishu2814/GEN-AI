"""
First LLM API Program using Groq

This program demonstrates:
1. Loading API key from .env
2. Creating an OpenAI-compatible client
3. Sending a prompt
4. Printing the response
5. Handling errors
"""
# Loads variables from the .env file
from dotenv import load_dotenv
# to read env variables 
import os
from openai import OpenAI

# ----------------------------------
# Load environment variables
# ----------------------------------
load_dotenv()

# ----------------------------------
# Read API Key
# ----------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ----------------------------------
# Create Groq Client
# ----------------------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.3-70b-versatile"

PROMPT = "What is Binary Search and explain in pointwise for my exam as 5 mark question."

try:

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ]
    )

    print("\n========== AI Response ==========\n")
    # print(response)
    print(response.choices[0].message.content)

except Exception as e:
    print("\n❌ Something went wrong!")
    print(e)
    
    

    