import os
from openai import OpenAI
from dotenv import load_dotenv
from openai import RateLimitError
import json
import time
import requests
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()  
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    #base_url=os.getenv("BASE_URL")
)


def get_weather(city : str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    
    if response.status_code == 200:
      return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"   

available_tools = {
    "get_weather": get_weather    
}

SYSTEM_PROMPT = """
You're an expert AI Assistance in resolving user querib es using chain og thoughts.
You work on START, PLAN and Output steps.
You need to first plan what needs to be done. The plan can multiple steps.
Once you think enough Plan has been done, finally you can give output.
You can also call tool if required from the list of available tools. 
For every tool call wait for observe step which is the output from called tool.

Rules: 
1. Strictly follow the given JSON output format.
2. Only run one step at a time.
3. The sequence of steps is START (Here user gives an input), Plan (that can be multiple times) and finally Output (Which is going to be displayed to the user).

Output JSON format:
{"step": "PLAN" | "TOOL" | "OUTPUT", "content": "string", "tool": "string", "input": "string"}

Available Tools:
-get_weather(city: str): Takes city name as input string and returns the weather info about the city.

Example: 
START: What is the weather of Delhi?
PLAN:{"step": "PLAN": "content": "Seems like user is interested in getting weather of Delhi in India"} 
PLAN:{"step": "PLAN": "content": "Lets see if we have any available tool from list to get the weather"} 
PLAN:{"step": "TOOL": "get_weather", "input":Delhi}
PLAN:{"step": "OBSERVE": "tool": "get_weather, "output":The weather if delhi is cloudy with 20 c}
OUTPUT:{"step": "OUTPUT": "content":"The current weather in delhi is 20"c with cloudy sky."}
 
"""

class MyOutput(BaseModel):
    step: str = Field(..., description="The id of the step. Example: PLAN, OUTPUT, TOOL etc")
    content: Optional[str] = Field(None, description = "The optional string content for step")
    tool: Optional[str] = Field(None, description = "The Id of the tool to call")
    input: Optional[str] = Field(None, description = "The input params for the tool")

message_history = [
    {"role": "system", "content":SYSTEM_PROMPT},
    
]

user_query = input(">")
message_history.append({"role": "user", "content": user_query})


while True:
    # Handle transient rate limit errors with retries/backoff
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.parse(
                model="gpt-4o",
                response_format=MyOutput,
                messages=message_history,
            )
            break
        except RateLimitError as e:
            wait = min(2 ** attempt * 0.5, 10)
            print(f"[RATE LIMIT] attempt {attempt+1}/{max_retries}, retrying in {wait} seconds...")
            time.sleep(wait)
    else:
        raise
    
    
    raw_result = response.choices[0].message.content
    message_history.append({"role": "assistant", "content": raw_result})
    
    try:
        parsed_result = response.choices[0].message.parsed
    except json.JSONDecodeError:
        print(f"**[JSON ERROR]** Could not parse JSON. Raw response: {raw_result}")
        break # Emergency exit
        
    step_data = parsed_result.step    
    # If step_data is a dictionary (like in your prompt example), extract the command
    if isinstance(step_data, dict):
        step_type = step_data.step
    else:
        step_type = str(step_data).upper()

    if step_type in ["START", "PLAN"]:
        print(f"[{step_type}]: {parsed_result.content}")
        continue
    
    # 2. Handle Tool Calls
    if step_type == "TOOL":
        tool_to_call = parsed_result.tool
        tool_input = parsed_result.input
        
        if tool_to_call in available_tools:
            print(f"-> Calling tool: {tool_to_call} ({tool_input})")
            tool_response = available_tools[tool_to_call](tool_input)
            
            # Feed result back as an 'OBSERVE' step
            message_history.append({
                "role": "user", 
                "content": json.dumps({"step": "OBSERVE", "tool": tool_to_call, "output": tool_response})
            })
        else:
            print(f"**[ERROR]** Tool '{tool_to_call}' not found.")
        continue
    
    # 3. Handle Final Output
    elif step_type == "OUTPUT":
        print(f"\n[FINAL ANSWER]: {parsed_result.content}")
        break