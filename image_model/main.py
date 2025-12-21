from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content":[
                {"type": "text", "text": "Generate caption for this image in about 10 words"},
                {"type": "image_url", "image_url": {"url":"https://i0.wp.com/blogs.embarcadero.com/wp-content/uploads/2022/11/pexels-hitesh-choudhary-879109-9236935-scaled.jpg?ssl=1"}}
            ]
        }
    ]    
)
print("Response:", response.choices[0].message.content)
