from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# r -> read mode, encoding="utf-8" to handle special characters in the resume
with open("3.Prompt/Krishu_14June_Resume.txt", "r", encoding="utf-8") as file:
    resume = file.read()

# print(resume)

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([(
    "system",
        """
        You are a resume reviewer.

        Return ONLY valid JSON.

        {{
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }}
        """
    ),
    (
       (
    "human",
    """
    Review the following resume.

    Resume:

    ```
    {resume}
    ```
    """
)
    )
])

chain = prompt | model

result = chain.invoke({"resume": resume})

print(result.content) # returns the JSON output as a string
