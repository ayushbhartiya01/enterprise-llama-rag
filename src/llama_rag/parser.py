from pathlib import Path
from llama_cloud import LlamaCloud
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_rag.config import settings


def load_and_parse_documents(data_dir: str):
    """Parse PDFs with the LlamaParse v2 API and split the Markdown into nodes."""
    pdf_paths = sorted(
        path for path in Path(data_dir).iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{data_dir}'.")

    client = LlamaCloud(api_key=settings.llama_cloud_api_key)
    print(f"[*] Uploading and parsing files inside '{data_dir}/' directory...")

    documents = []
    for pdf_path in pdf_paths:
        uploaded_file = client.files.create(file=str(pdf_path), purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded_file.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )
        markdown = "\n\n".join(
            page.markdown
            for page in result.markdown.pages
            if page.markdown
        )
        if markdown.strip():
            documents.append(
                Document(text=markdown, metadata={"file_name": pdf_path.name})
            )

    if not documents:
        raise ValueError(f"LlamaParse returned no Markdown for PDFs in '{data_dir}'.")

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
