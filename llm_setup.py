from crewai import LLM
import os
import requests
from dotenv import load_dotenv


LOCAL_URL="http://127.0.0.1:11434"
TEMPERATURE=0

def get_llm():
    """
    Initializes and returns LLM instance.
    Priority:
    1. If Ollama(local) is running, use that (default model: mistral).
    2. Otherwise, fall back to Gemini (** NEEDS API KEY **)
    """
    load_dotenv()
    try:
        requests.get(LOCAL_URL, timeout=1)
        base_url=os.getenv("OLLAMA_BASE_URL",LOCAL_URL)
        model=os.getenv("OLLAMA_CHAT_MODEL","mistral")
        if "/" not in model:
            model=f"ollama/{model}"

            llm= LLM(base_url=base_url,
                    model=model,
                    temperature=TEMPERATURE)
    except requests.exceptions.RequestException:
        llm= LLM(
            model="gemini/gemini-1.5-flash",
            temperature=TEMPERATURE
            )
    return llm