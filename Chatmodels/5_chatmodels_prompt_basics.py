from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template(
"""
You are a {profession}.

Explain {topic}

using {language} language.
"""
)

final_prompt = prompt.invoke({
        "profession": "Backend Engineer",
        "topic": "HashMap",
        "language": "Java"
    });
print(type(final_prompt)) # <class 'langchain_core.prompt_values.StringPromptValue'>
result= model.invoke(final_prompt)
print(type(result)) # <class 'langchain_core.messages.ai.AIMessage'>
# print(result.content);
