from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

@tool
def weather(city: str) -> str:
    """
    Returns the current weather of a city.
    """

    return f"The weather in {city} is 31°C and Sunny."


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """

    try:
        result = eval(expression)
        return f"Answer = {result}"

    except Exception:
        return "Invalid mathematical expression."


@tool
def greet(name: str) -> str:
    """
    Greets a user.
    """

    return f"Hello {name}! Welcome to LangChain."


# =========================================================
# Register All Tools with the LLM
# =========================================================

tools = [
    weather,
    calculator,
    greet
]

llm_with_tools = llm.bind_tools(tools)

# =========================================================
# Tool Registry
# Maps Tool Name ---> Tool Object
# =========================================================

tool_registry = {
    tool.name: tool
    for tool in tools
}

# Same as:
#
# tool_registry = {
#     "weather": weather,
#     "calculator": calculator,
#     "greet": greet
# }

print("\nRegistered Tools:")
print(tool_registry.keys())

# =========================================================
# User Input
# =========================================================

messages = [
    HumanMessage(
        content="weather of ranchi"
    )
]

# Try these:
#
# "What's the weather in Delhi?"
#
# "Say hello to Krishu"
#
# "Calculate (250 + 75) / 5"

# =========================================================
# First LLM Call
# =========================================================

response = llm_with_tools.invoke(messages)

messages.append(response)

print("\n" + "=" * 60)
print("FIRST LLM RESPONSE")
print("=" * 60)

print(response)

# =========================================================
# Execute Tool Dynamically
# =========================================================

if response.tool_calls:

    tool_call = response.tool_calls[0]

    tool_name = tool_call["name"]

    tool_args = tool_call["args"]

    tool_id = tool_call["id"]

    print("\nTool Requested :", tool_name)

    print("Arguments      :", tool_args)

    # -----------------------------------------------------
    # Dynamic Tool Lookup
    # No if-elif Required
    # -----------------------------------------------------

    selected_tool = tool_registry.get(tool_name)

    if selected_tool is None:

        print("Unknown Tool!")

    else:

        tool_result = selected_tool.invoke(tool_args)

        print("\n" + "=" * 60)
        print("TOOL RESULT")
        print("=" * 60)

        print(tool_result)

        # -------------------------------------------------
        # Create Tool Message
        # -------------------------------------------------

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_id
        )

        messages.append(tool_message)

        # -------------------------------------------------
        # Second LLM Call
        # -------------------------------------------------

        final_response = llm_with_tools.invoke(messages)

        print("\n" + "=" * 60)
        print("FINAL RESPONSE")
        print("=" * 60)

        print(final_response.content)

else:

    print("No Tool Called.")
    
    