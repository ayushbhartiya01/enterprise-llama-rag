from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_rag.config import settings

def get_qdrant_client() -> QdrantClient:
    """Returns a unified Qdrant client connection."""
    return QdrantClient(url=settings.qdrant_url)

def get_vector_store() -> QdrantVectorStore:
    """Initializes and returns the Qdrant Vector Store plugin, creating the collection if missing."""
    client = get_qdrant_client()
    collection_name = "enterprise_collection"
    
    # Auto-provision the collection if it doesn't exist yet
    if not client.collection_exists(collection_name):
        print(f"[*] Collection '{collection_name}' not found. Provisioning structure...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,  # Crucial match for BAAI/bge-large-en-v1.5 text embeddings
                distance=Distance.COSINE
            )
        )
        print(f"[+] Collection '{collection_name}' created successfully.")
        
    return QdrantVectorStore(client=client, collection_name=collection_name)
