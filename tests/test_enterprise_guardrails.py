import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from llama_rag.guardrails import (
    EnterpriseGroundingGuard,
    GuardrailViolationError,
    reject_broad_summary_request,
)


def test_grounding_guard_accepts_a_score_at_the_threshold():
    node = NodeWithScore(node=TextNode(text="Relevant passage"), score=0.6)

    assert EnterpriseGroundingGuard(threshold=0.6)._postprocess_nodes([node]) == [node]


def test_grounding_guard_explains_low_relevance_score():
    node = NodeWithScore(node=TextNode(text="Relevant passage"), score=0.42)

    with pytest.raises(
        GuardrailViolationError,
        match=r"relevance score \(0\.420\).+threshold \(0\.600\)",
    ):
        EnterpriseGroundingGuard(threshold=0.6)._postprocess_nodes([node])


def test_grounding_guard_explains_empty_reranker_results():
    with pytest.raises(GuardrailViolationError, match="No document passages"):
        EnterpriseGroundingGuard(threshold=0.6)._postprocess_nodes([])


@pytest.mark.parametrize(
    "query",
    [
        "Summarize the document's main ideas.",
        "Summarizing all PDFs",
    ],
)
def test_broad_document_summary_is_rejected(query):
    with pytest.raises(GuardrailViolationError, match="Broad document or corpus summaries"):
        reject_broad_summary_request(query)


def test_focused_section_summary_is_allowed():
    reject_broad_summary_request("Summarize the section about self-attention.")


def test_focused_question_is_not_treated_as_a_summary():
    reject_broad_summary_request("What is a Transformer?")
