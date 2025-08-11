# %%
import torch
import gc
import time

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,BitsAndBytesConfig
BASE_PATH = Path("/mnt/sdg/tzhu/llm")
TRANSCRIPT_PATH = f"{BASE_PATH}/psychs_transcripts"

# %%
def generate_assessment(prompt, llama_model, llama_tokenizer):
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
    
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=1024,
            # temperature=0.3,
            # do_sample=True,
            pad_token_id=llama_tokenizer.eos_token_id
        )
    
    # Decode and get only generated part
    assessment = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # assessment = assessment[len(prompt):].strip()
    return assessment


# %%
# -------- LLaMA 4 Scout 17B-16E --------
llama_model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

llama_local_path = f"{BASE_PATH}/{llama_model_id}"

llama_tokenizer = AutoTokenizer.from_pretrained(
    llama_local_path, 
    local_files_only=True
)

max_m = 35
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_local_path,  
    device_map='balanced_low_0',  # Prioritize GPU 0, offload to CPU when needed
    max_memory={
        0: f"{max_m}GiB",  # GPU 0
        1: f"{max_m}GiB",  # GPU 1  
        2: f"{max_m}GiB",  # GPU 2
        3: f"{max_m}GiB",  # GPU 3
        4: f"{max_m}GiB",  # GPU 4
        "cpu": "200GiB"  # CPU RAM
    },
    
    # quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True  
)

# %%
with open('prompt.txt', 'r') as f:
    prompt_template = f.read()
    
transcript_files = [f for f in Path(TRANSCRIPT_PATH).glob("*.txt") 
                   if "prompt" not in f.name and "assessment" not in f.name]


for transcript_file in transcript_files:
    # Read transcript
    with open(transcript_file, 'r') as f:
        transcript = f.read()
    
    # Create prompt with transcript
    prompt = prompt_template.format(transcript=transcript)
    
    # Generate assessment
    print(f"\nProcessing: {transcript_file.name}")
    gen_start = time.time()
    assessment = generate_assessment(prompt, llama_model, llama_tokenizer)
    gen_time = time.time() - gen_start
    print(f"Generation time: {gen_time:.2f}s")
    # Save assessment
    output_file = str(transcript_file).replace('.txt', '_assessment.txt')
    with open(f'{output_file}', 'w') as f:
        f.write(f"File: {transcript_file.name}\n")
        f.write("="*50 + "\n")
        f.write(assessment)
    
    print(f"Saved: {output_file}")

    # CLEAR MEMORY AFTER EACH FILE
    torch.cuda.empty_cache()
    gc.collect()
