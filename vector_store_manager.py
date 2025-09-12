import chromadb
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class VectorStoreManager:
    def __init__(self,collection_name="repo_intellect_store"):
        self.client=chromadb.Client()
        self.collection_name= collection_name
        self.embedding_function=OllamaEmbeddings(model="mistral")
        self.vector_store=self.get_or_create_vector_store()

    def get_or_create_vector_store(self):
        """
        Initializes the Chroma vector store.
        Creates a new collection if it doesn't exist.
        """
        print(f"Initializing vector store with collection:{self.collection_name}")
        return Chroma(client=self.client,
                      collection_name=self.collection_name,
                      embedding_function=self.embedding_function
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