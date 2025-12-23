# STT Model-   https://pypi.org/project/SpeechRecognition/       
# TTS Model-   https://platform.openai.com/docs/guides/text-to-speech  
# AttributeError fix the issue command brew install portaudio. 
# Then install with PyAudio using Pip: pip install "SpeechRecognition[audio]"
# This feature requires additional dependencies pip install "openai[voice_helpers]"

import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
from openai.helpers import LocalAudioPlayer


load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()



async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=speech,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


def main():
    r = sr.Recognizer() #Speech to text
    
    
    while True:
        with sr.Microphone() as source: #Mic Access
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2
            
            SYSTEM_PROMPT= """
                You're an expert voice agent, You are given transcript of what user has said using voice. 
                You need to output as if you an voice agent and whatever you speak is going to be converted back to audio using AI and played back to user. 
            """
            
            messages = [                    
                {"role": "system", "content" : SYSTEM_PROMPT},
            ]
            
            print("Speak Something...")
            audio = r.listen(source)
            
            print("Processing Audio..(Speech to Text)") 
            stt = r.recognize_google(audio)
            
            print("You ask", stt)
            
            
            messages.append({"role": "user", "content" : stt})
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
            print("AI Response", response.choices[0].message.content)
            asyncio.run(tts(speech=response.choices[0].message.content))
main()