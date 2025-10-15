import os
import requests
import torch
import json, re, time
from urllib.parse import urljoin
from pathlib import Path

TRANSCRIPT_PATH = "psychs_transcripts"
MODEL_NAME = "medgemma-27b-text-it"
RETRY_WAIT = 1800

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", f"/scratch/users/{os.getenv('USER','user')}/hf_cache")

MODEL_PATH = os.environ.get("MEDGEMMA_LOCAL_PATH", f"/scratch/users/k2585724/models/{MODEL_NAME}")


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"[INFO] Loading model from: {MODEL_PATH}")
print(f"[INFO] Device: {device}, DType: {dtype}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    local_files_only=True,
)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "How do you differentiate bacterial from viral pneumonia?"}
]

if hasattr(tokenizer, "apply_chat_template"):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
else:
    prompt = f"<system>\n{messages[0]['content']}\n</system>\n<user>\n{messages[1]['content']}\n</user>\n<assistant>\n"
    inputs = tokenizer(prompt, return_tensors="pt")


inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

gen_kwargs = dict(max_new_tokens=256, do_sample=False, eos_token_id=tokenizer.eos_token_id)

while True:
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    if output_ids.dim() == 2:
        output_ids = output_ids[0]
    if hasattr(tokenizer, "apply_chat_template"):
        output_ids = output_ids[input_len:]

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\n================== MODEL OUTPUT ==================\n")
    print(text)
    print("\n==================================================\n")
    time.sleep(RETRY_WAIT)
