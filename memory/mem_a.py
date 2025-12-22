#This code will use OpenAI text-embedding model and convert user query and vector and store into qudrant db locally.

from mem0 import Memory
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {
    "version": "v1.1",
    "embedder":{
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}
    },
    "llm":{
      "provider": "openai",
      "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4.1"}  
    },
    "vector_store":{
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}
mem_client = Memory.from_config(config)


while True:
    user_query = input(">")
    search_memory = mem_client.search(query=user_query, user_id="Raj",)
    
    memories = [
        f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}"
        for mem in search_memory.get("results")
    ]
    print("Found Memories", memories)
    
    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    ai_response = response.choices[0].message.content
    
    print("AI:", ai_response)
    
    mem_client.add(
        user_id="Raj",
        messages=[
            {"role": "user", "content": user_query},
            {"role": "user", "content": ai_response},            
        ]
    )
    print("Memory has been updated...")    