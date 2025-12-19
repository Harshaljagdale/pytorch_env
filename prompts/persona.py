import os
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()  

client = OpenAI(
    api_key=os.getenv("OPEN_API_KEY"),
    #base_url=os.getenv("BASE_URL")
)

SYSTEM_PROMPT = """
You are an AI Persona Assiatance named Alexa
You are acting on behalf of Raj whos is 30 years and Tech enthusiatic and 
principle engineer. Your main tech slack is Salesforce and Python and you are 

"""


response = client.chat.completions.create(
        model="gpt-4o", 
        messages = [
            {"role": "system", "content":SYSTEM_PROMPT},
            {"role": "user", "content":"Hey There"}
        ]
    )

print("Response:", response.choices[0].message.content) 
