from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from PIL import Image
import requests

# Load the model (choose one based on your hardware)
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"  # Smaller 7B version
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"  # Larger 13B version

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(device)

# Prepare image and prompt
image_url = "https://llava-vl.github.io/static/images/view.jpg"  # Example image
image = Image.open(requests.get(image_url, stream=True).raw)

prompt = "USER: <image>\nWhat's in this image?\nASSISTANT:"

# Process and generate
inputs = processor(prompt, image, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0][2:], skip_special_tokens=True))
