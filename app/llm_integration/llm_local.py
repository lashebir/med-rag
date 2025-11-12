# llm_local.py
import requests, json

def generate_local(prompt: str, context_chunks: list[str]) -> str:
    sys = ("You are a biomedical assistant. Use ONLY the provided context. "
           "Cite sources like [PMCID:chunk]. If unsure, say you don’t know.")
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks))
    full = f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n[CONTEXT]\n{ctx}\n[/CONTEXT]\n[USER]\n{prompt}\n[/USER]\n"
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b-instruct-q4_K_M",
        "prompt": full,
        "stream": False,
        "options": {"temperature": 0.2}
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"]
