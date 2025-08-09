# %%
import torch
import os
import shutil
import nemo.collections.asr as nemo_asr 
import soundfile as sf
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio

BASE_PATH = Path("/mnt/sdg/tzhu/llm")
os.makedirs(BASE_PATH, exist_ok=True)

# %%
# -------- LLaMA 3 --------
llama_model_id = "meta-llama/Llama-3.2-3B-Instruct"

llama_local_path = os.path.join(BASE_PATH, llama_model_id)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id, cache_dir=llama_local_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, 
                                                   device_map='auto',
                                                   cache_dir=llama_local_path, 
                                                   trust_remote_code=True, 
                                                   torch_dtype=torch.float16)

# Test LLaMA
llama_pipe = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)
print("LLaMA test:")
print(llama_pipe("Summarise this candidate's strengths based on the interview notes: ", max_new_tokens=50)[0]["generated_text"])




