from crewai import LLM
import os


def get_llm():
    """
    Initializes and returns ChatOllama model instance.
    This assumes Ollama is running local with "mistral" model.
    """
    base_url=os.getenv("OLLAMA_BASE_URL","http://127.0.0.1:11434")
    model=os.getenv("OLLAMA_CHAT_MODEL","mistral")
    if "/" not in model:
        model=f"ollama/{model}"

    llm= LLM(base_url=base_url,
                    model=model,
                    temperature=0)
    return llm