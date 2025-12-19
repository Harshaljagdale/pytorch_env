import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from openai import RateLimitError
load_dotenv()  

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)
SYSTEM_PROMPT = """
You're an expert AI Assiatance in resolving user quries using chain og thoughts.
You work on START, PLAN and Output steps.
You nees to first plan what needs to be done. The plan can multiple steps.
Once you think enough Plan has been done, finally you can give output.

Rules: 
1. Strickly follow the given JOSN output format.
2. Only run one step at a time.
3. The sequence of stpes is START (Hwere user gives an input), Plan (that can be multiple times) and finally Output (Which is going to be displayed to the user).

Output JSON format:
{"steps":START | "Plan" | "Output", "content": "string"}

"""

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
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                response_format={"type": "json_object"},
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
        parsed_result = json.loads(raw_result)
    except json.JSONDecodeError:
        print(f"**[JSON ERROR]** Could not parse JSON. Raw response: {raw_result}")
        break # Emergency exit
        
    step_value = parsed_result.get("steps", "").upper()
    
    if step_value == "START" or step_value == "PLAN":
        print(parsed_result.get("content"))
        continue
    elif step_value == "OUTPUT":
        print(parsed_result.get("content"))
        break
    else:
        print(f"**[FATAL ERROR]** Unexpected step value: {parsed_result.get('steps')}")
        print(parsed_result)
        break