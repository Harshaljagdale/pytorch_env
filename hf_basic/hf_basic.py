from transformers import pipeline, AutoProcessor
import torch

model_id = "google/gemma-3-4b-it"

# 1. Load the processor separately to handle the pad_token
processor = AutoProcessor.from_pretrained(model_id)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# 2. Initialize pipeline with the processor
pipe = pipeline(
    "image-text-to-text", 
    model=model_id,
    device="mps", # Use "mps" for Mac GPU acceleration
    torch_dtype=torch.float16,
    feature_extractor=processor.image_processor,
    tokenizer=processor.tokenizer
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]

# 3. Add padding to the call
outputs = pipe(
    messages, 
    generate_kwargs={"max_new_tokens": 100},
    padding=True  # This tells the pipeline to fix the length issues
)

print(outputs[0]["generated_text"])