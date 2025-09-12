from langchain_ollama.chat_models import ChatOllama

def get_llm():
    """
    Initializes and returns ChatOllama model instance.
    This assumes Ollama is running local with "mistral" model.
    """

    llm= ChatOllama(model="mistral", temperature=0)
    return llm