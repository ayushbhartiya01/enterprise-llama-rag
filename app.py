import os
import sys
# from llama_index.core.base.exceptions import ValidationError
# from pydantic import ValidationError 
from llama_rag.guardrails import GuardrailViolationError, reject_broad_summary_request
from llama_rag import load_and_parse_documents, build_production_query_engine

def main():
    data_dir = "data"
    
    # Check if a documents folder exists
    if not os.path.exists(data_dir):
        print(f"[*] Creating '{data_dir}/' directory. Place your enterprise PDFs there.")
        os.makedirs(data_dir)
        sys.exit(0)

    # Simple interactive CLI flag to trigger initial ingestion
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        print("[*] Starting LlamaParse Ingestion Worker...")
        nodes = load_and_parse_documents(data_dir)
        print(f"[+] Successfully extracted {len(nodes)} structural Markdown nodes.")
        
        print("[*] Re-indexing and upserting data into Qdrant Cluster...")
        query_engine = build_production_query_engine(nodes=nodes)
        print("[+] Vector storage sync complete.")
    else:
        print("[*] Connecting to existing Qdrant index...")
        query_engine = build_production_query_engine(nodes=None)

    print("\nEnterprise LlamaIndex RAG Online. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nEnterprise User Request > ").strip()
            if not query or query.lower() == "exit":
                break

            reject_broad_summary_request(query)
            response = query_engine.query(query)
            print(f"\n[LLM Response]:\n{response}")
            
        except GuardrailViolationError as guard_err:
            # Safely trap corporate data leaks or injection mitigations here
            print(f"\n[Guardrail Alert]: {guard_err}")
        except Exception as e:
            print(f"\n[System Error]: {str(e)}")

if __name__ == "__main__":
    main()
