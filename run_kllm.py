# %%
import os
import requests
import json
import re
import time
import argparse
from urllib.parse import urljoin
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
BASE = os.getenv("OPENWEBUI_BASE", "https://ai.create.kcl.ac.uk")
TOKEN = os.getenv("OPENWEBUI_TOKEN", "sk-67123084b3384c1a8f70461dd9c60c5b")
TIMEOUT = 120
TRANSCRIPT_PATH = "psychs_transcripts"
OUTPUT_DIR = Path("outputs_score")
MAX_API_RETRIES = 3         # Retries for API calls
RETRY_WAIT = 5              # seconds between retries

_session = requests.Session()


# -------------------------------
# Helper functions
# -------------------------------
def _headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _url(path: str) -> str:
    return urljoin(BASE.rstrip("/") + "/", path.lstrip("/"))

def _parse_response_text(txt: str, ctype: str):
    if not txt.strip():
        raise RuntimeError("Empty response body (server returned no JSON).")

    if txt.lstrip().startswith(("{", "[")):
        return json.loads(txt)

    if "text/event-stream" in ctype or "data:" in txt:
        chunks = []
        for line in txt.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                obj = json.loads(line[5:].strip())
            except Exception:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            piece = delta.get("content")
            if piece is None:
                msg = ch.get("message") or {}
                piece = msg.get("content")
            if piece:
                chunks.append(piece)
        if chunks:
            return {"choices": [{"message": {"content": "".join(chunks)}}]}

    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        return json.loads(m.group(0))

    raise RuntimeError(f"Unexpected response (first 300 chars): {txt[:300]}")

def _post_json(path: str, payload: dict):
    for attempt in range(MAX_API_RETRIES):
        try:
            r = _session.post(
                _url(path),
                headers=_headers(),
                json=payload,
                timeout=(10, 300),
            )

            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} {_url(path)}\nResponse: {r.text[:200]}")

            ctype = (r.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                return r.json()
            return _parse_response_text(r.text, ctype)

        except Exception as e:
            print(f"[API Retry {attempt+1}/{MAX_API_RETRIES}] {e}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            else:
                raise

def _extract_assistant_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        try:
            return data["choices"][0]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected response structure: {data}")

def chat(msg: str, model: str) -> str:
    payload = {"model": model, "messages": [{"role": "user", "content": msg}], "stream": False}
    for path in ("/api/chat/completions", "/v1/chat/completions"):
        try:
            data = _post_json(path, payload)
            return _extract_assistant_text(data)
        except Exception as e:
            print(f"Endpoint {path} failed: {e}")
    raise RuntimeError("Both API endpoints failed.")


# -------------------------------
# Main script
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model assessments on transcripts.")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., 'qwen3'")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt template name")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = Path(f"prompts/prompt_{args.prompt}.txt")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    transcript_files = [f for f in Path(TRANSCRIPT_PATH).glob("*.txt")
                        if "prompt" not in f.name and "assessment" not in f.name]

    for transcript_file in transcript_files:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        prompt = prompt_template.format(transcript=transcript)
        print(f"\n📄 Processing: {transcript_file.name} with model {args.model}...")

        success = False
        attempt = 0
        while not success:
            attempt += 1
            try:
                assessment = chat(prompt, args.model)
                success = True
            except Exception as e:
                print(f"⚠️ Attempt {attempt}: Failed ({e}) — retrying in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)

                # Optional: stop if stuck too long (avoid infinite loop)
                if attempt >= 20:
                    print(f"❌ Giving up on {transcript_file.name} after {attempt} tries.")
                    error_path = OUTPUT_DIR / f"{transcript_file.stem}_assessment_{args.prompt}_{args.model}_ERROR.txt"
                    with open(error_path, "w", encoding="utf-8") as ef:
                        ef.write(f"File: {transcript_file.name}\n{'='*50}\nError after {attempt} attempts:\n{e}")
                    break

        if not success:
            continue

        output_file = OUTPUT_DIR / f"{transcript_file.stem}_assessment_{args.prompt}_{args.model}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"File: {transcript_file.name}\n{'='*50}\n{assessment}")
        print(f"✅ Saved: {output_file}")