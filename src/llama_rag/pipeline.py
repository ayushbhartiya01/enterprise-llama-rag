from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_rag.config import settings
from llama_rag.database import get_vector_store
from llama_rag.guardrails import EnterpriseGroundingGuard

def build_production_query_engine(nodes=None) -> RetrieverQueryEngine:
    # 1. Global component registration
    LlamaSettings.llm = Groq(model=settings.llm_model, api_key=settings.groq_api_key)
    LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_name)
    
    # 2. Database client provisioning
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. Create or point to the index
    if nodes:
        index = VectorStoreIndex(nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    # 4. Advanced Two-Stage Retriever Definition
    # Stage 1: Broad semantic retrieval fetching a wide candidate pool (dense embedding space)
    base_retriever = index.as_retriever(similarity_top_k=settings.similarity_top_k)
    
    # Stage 2: Cross-Encoder heavy re-ranking to prioritize contextual alignment
    cohere_rerank = CohereRerank(
        model=settings.rerank_model,
        api_key=settings.cohere_api_key, 
        top_n=settings.rerank_top_n
    )
    
    # Stage 3: Dynamic Anti-hallucination processing
    grounding_guard = EnterpriseGroundingGuard(threshold=settings.grounding_threshold)
    
    # 5. Build optimized runtime query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        node_postprocessors=[cohere_rerank, grounding_guard]
    )
    
    return query_engine
