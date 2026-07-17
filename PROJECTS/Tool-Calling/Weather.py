from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage

# load environment variables
load_dotenv()

# model
model = ChatGroq(model="llama-3.3-70b-versatile")

# create tool
@tool
def get_weather(city : str) -> str :
    """
    Returns current weather for a city.
    """
    # Imagine this is calling a real Weather API
    return f"The current weather in {city} is 31°C and Sunny."

    
# bind tool
model_with_tools=model.bind_tools([get_weather])

# prompt (user) -> 1
prompt_first=[
    HumanMessage(content="WHat is the weather condition of Bihar?")]

# LLM call -> 1
result_first=model_with_tools.invoke(prompt_first)
print(result_first)

# check for tools calls
if result_first.tool_calls :
    tool_=result_first.tool_calls[0]
    print(tool_)
    tool_name=tool_['name']
    tool_args=tool_['args']
    tool_id=tool_['id']
    print(tool_args)
    print(tool_name)
    
    # execute tool call
    tool_result= get_weather.invoke(tool_args)
    print("*"*50 )
    print("Tool response is -------->>>>>>>>>")
    print(tool_result)
    
    # create tool message
    tool_message=ToolMessage(content=tool_result, tool_call_id=tool_id)
    print(tool_message)
    
    # send to llm for 2nd call 
    # error ->  Must be a PromptValue, str, or list of BaseMessages.
    result_final=model_with_tools.invoke(
        # Human -> AI -> Tool -> AI(if any)
        [*prompt_first, result_first, tool_message]
    )
    
    print("*"*50 )
    print("Final Response of LLM is -------->>>>>>>>>")
    print("Output is: ",result_final.content)
    
    
else :
    print("No tools called!")

