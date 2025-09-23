import os
import requests
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

LOCAL_OLLAMA_URL=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

def _ollama_alive(base_url:str) -> bool:
    try:
        requests.post(f"{base_url}/api/tags",timeout=1)
        return True
    except requests.exceptions.RequestException:
        return False
    
def make_embedding():
    """
    Provider selection:
    - If EMBEDDINGS_PROVIDER=ollama -> require Ollama alive or raise
    - If EMBEDDINGS_PROVIDER=gemini -> always use Gemini
    - If EMBEDDINGS_PROVIDER=auto (default) -> try Ollama, else Gemini
    """

    provider= os.getenv("EMBEDDINGS_PROVIDER","auto").lower()

    if provider in("ollama","auto") and _ollama_alive(LOCAL_OLLAMA_URL):
        return OllamaEmbeddings(
            base_url=LOCAL_OLLAMA_URL,
            model=os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text")
        )

    if provider== "ollama":
        raise RuntimeError ("EMBEDDINGS_PROVIDER = ollama but Ollama is not reachable at"
                            f"{LOCAL_OLLAMA_URL}. Start 'ollama server' or change provider")
    
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL","models/text-embedding-004"),
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
class VectorStoreManager:
    def __init__(self,collection_name="repo_intellect_store"):
        self.client=chromadb.PersistentClient("./chroma_db")
        self.collection_name= collection_name
        self.embedding_function=make_embedding()
        self.vector_store=self.get_or_create_vector_store()

    def get_or_create_vector_store(self):
        """
        Initializes the Chroma vector store.
        Creates a new collection if it doesn't exist.
        """
        print(f"Initializing vector store with collection:{self.collection_name}")
        return Chroma(client=self.client,
                      collection_name=self.collection_name,
                      embedding_function=self.embedding_function,
                      )
    
    def populate_vector_store(self, documents:list):
        """
        Adds documents to the vector store.
        """
        if not documents:
            print("No documents to add to the store.")
            return

        print(f"Adding {len(documents)} document chunks to the vectore store...")
        self.vector_store.add_documents(documents=documents)
        print("Documents added successfully")

    def get_retriever(self, search_kwargs={"k":5}):
        """
        Returns a retriever instance of the vector store
        """
        print(f"Creating a retriever with search_kwargs:{search_kwargs}")
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)