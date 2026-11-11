from pathlib import Path
from types import SimpleNamespace

import pytest

import llama_rag.parser as parser_module


class FakeLlamaCloud:
    def __init__(self, *, api_key):
        assert api_key == "test-llama-cloud-key"
        self.files = SimpleNamespace(create=self.create_file)
        self.parsing = SimpleNamespace(parse=self.parse_file)
        self.uploaded_paths = []
        self.parse_requests = []

    def create_file(self, *, file, purpose):
        assert purpose == "parse"
        self.uploaded_paths.append(Path(file).name)
        return SimpleNamespace(id=f"id-{Path(file).name}")

    def parse_file(self, **kwargs):
        self.parse_requests.append(kwargs)
        return SimpleNamespace(
            markdown=SimpleNamespace(
                pages=[
                    SimpleNamespace(markdown="# Policy\nRetain records for seven years."),
                    SimpleNamespace(markdown=""),
                    SimpleNamespace(markdown="Contact the records team."),
                ]
            )
        )


def test_load_and_parse_documents_uploads_pdfs_and_preserves_source(monkeypatch, tmp_path):
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"placeholder PDF")
    (tmp_path / "notes.txt").write_text("not a PDF")
    client = FakeLlamaCloud(api_key="test-llama-cloud-key")
    monkeypatch.setattr(parser_module, "LlamaCloud", lambda **kwargs: client)

    nodes = parser_module.load_and_parse_documents(str(tmp_path))

    assert client.uploaded_paths == ["policy.pdf"]
    assert client.parse_requests == [
        {
            "file_id": "id-policy.pdf",
            "tier": "agentic",
            "version": "latest",
            "expand": ["markdown"],
        }
    ]
    assert len(nodes) == 1
    assert "Retain records for seven years." in nodes[0].text
    assert "Contact the records team." in nodes[0].text
    assert nodes[0].metadata["file_name"] == "policy.pdf"


def test_load_and_parse_documents_raises_when_no_pdfs_exist(tmp_path):
    (tmp_path / "notes.txt").write_text("not a PDF")

    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        parser_module.load_and_parse_documents(str(tmp_path))


def test_load_and_parse_documents_raises_when_parse_returns_no_markdown(
    monkeypatch, tmp_path
):
    (tmp_path / "empty.pdf").write_bytes(b"placeholder PDF")

    class EmptyMarkdownClient:
        def __init__(self, *, api_key):
            self.files = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(id="empty-file")
            )
            self.parsing = SimpleNamespace(
                parse=lambda **kwargs: SimpleNamespace(
                    markdown=SimpleNamespace(pages=[SimpleNamespace(markdown=" ")])
                )
            )

    monkeypatch.setattr(parser_module, "LlamaCloud", EmptyMarkdownClient)

    with pytest.raises(ValueError, match="returned no Markdown"):
        parser_module.load_and_parse_documents(str(tmp_path))
