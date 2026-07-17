from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

# ============================================================
# Load Environment Variables
# ============================================================
load_dotenv()

# ============================================================
# Create LLM
# ============================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ============================================================
# Create Tool
# ============================================================

@tool
def get_prime_minister(country: str) -> str:
    """ Return the information of prime minister of india. """
    if country.lower() == "india":
        return """
            Current Prime Minister of india is ->  Narendra Modi
        """

    return f"Prime Minister information for '{country}' is not available."

# ============================================================
# Bind Tool -> list
# ============================================================

llm_with_tools = llm.bind_tools([get_prime_minister])

# ============================================================
# User Message
# ============================================================

messages = [
    HumanMessage(
        content="Who is the Prime Minister of India? and what is the age of pm."
    )
]

# ============================================================
# First LLM Call
# ============================================================

response = llm_with_tools.invoke(messages)

print("=" * 60)
print("FIRST LLM RESPONSE")
print("=" * 60)
# print(response)

# Store AIMessage
messages.append(response)

# ============================================================
# Execute Tool (if requested)
# ============================================================

if response.tool_calls:

    tool_call = response.tool_calls[0]

    print("\nTool Selected :", tool_call["name"])
    print("Arguments     :", tool_call["args"])

    tool_result = get_prime_minister.invoke(tool_call["args"])

    print("\n" + "=" * 60)
    print("TOOL RESULT")
    print("=" * 60)
    # print(tool_result)

    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call["id"]
    )

    messages.append(tool_message)

    # ========================================================
    # Second LLM Call
    # ========================================================

    final_response = llm_with_tools.invoke(messages)

    print("\n" + "=" * 60)
    print("FINAL LLM RESPONSE")
    print("=" * 60)
    print(final_response.content)

else:

    print("No tool was called.")
    
    