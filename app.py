from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch

# --------------------------------------------------------------
# Model setup
# --------------------------------------------------------------
base_model_name = "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit"
adapter_name = "michsethowusu/opani-coder"

print("🔹 Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("✅ Model ready!")

# --------------------------------------------------------------
# API setup
# --------------------------------------------------------------
app = FastAPI()

# Allow Chat UI frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}

