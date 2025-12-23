import os
import json
import time
import asyncio
import requests
import speech_recognition as sr
from openai import OpenAI, AsyncOpenAI, RateLimitError
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional


try:
    from openai.helpers import LocalAudioPlayer
except ImportError:
    LocalAudioPlayer = None

load_dotenv()  
async_client = AsyncOpenAI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

r = sr.Recognizer()

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text}"
    except:
        pass
    return "I couldn't retrieve the weather right now."   

available_tools = {"get_weather": get_weather}

SYSTEM_PROMPT = """
You're an expert AI Assistant. Follow this loop: PLAN -> TOOL -> OBSERVE -> OUTPUT.
Rules:
1. Output ONLY JSON.
2. One step at a time.
3. If you have enough info, provide the OUTPUT step.

Output JSON format:
{"step": "PLAN", "content": "thinking process"}
{"step": "TOOL", "tool": "get_weather", "input": "cityname"}
{"step": "OUTPUT", "content": "final answer"}
"""

class MyOutput(BaseModel):
    step: str
    content: Optional[str] = None
    tool: Optional[str] = None
    input: Optional[str] = None

async def tts(speech: str):
    try:
        async with async_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=speech,
            response_format="pcm",
        ) as response:
            if LocalAudioPlayer:
                await LocalAudioPlayer().play(response)
            else:
                print(f"Assistant: {speech}")
    except Exception as e:
        print(f"TTS Error: {e}")


print("--- Voice Assistant Started (Say 'Exit' to stop) ---")

while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("\nListening...")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            user_query = r.recognize_google(audio)
            print(f"You: {user_query}")
            
            
            if any(word in user_query.lower() for word in ["exit", "stop", "quit", "terminate"]):
                print("Shutting down...")
                break
                
        except sr.WaitTimeoutError:
            continue 
        except Exception as e:
            print(f"Could not hear you: {e}")
            continue

    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    while True:
        try:
            response = client.chat.completions.parse(
                model="gpt-4o-mini",
                response_format=MyOutput,
                messages=message_history,
            )
        except RateLimitError:
            print("Rate limit hit, waiting...")
            time.sleep(5)
            continue

        res = response.choices[0].message.parsed
        message_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
        step_type = res.step.upper()

        if step_type == "PLAN":
            print(f"[Thinking]: {res.content}")
            continue
        
        elif step_type == "TOOL":
            if res.tool in available_tools:
                print(f"[Tool]: Calling {res.tool}...")
                result = available_tools[res.tool](res.input)
                message_history.append({
                    "role": "user", 
                    "content": json.dumps({"step": "OBSERVE", "output": result})
                })
            continue
        
        elif step_type == "OUTPUT":
            print(f"Assistant: {res.content}")
            asyncio.run(tts(res.content))
            break