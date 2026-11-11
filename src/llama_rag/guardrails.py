import re
from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


def reject_broad_summary_request(query: str) -> None:
    summary_intent = re.search(
        r"\b(?:summar(?:y|ies|ize|ise|izing|ising|ization|isation)|overview)\b",
        query,
        flags=re.IGNORECASE,
    )
    broad_target = re.search(
        r"\b(?:document|documents|pdfs?|files?|corpus|everything|"
        r"all|entire|whole|main ideas|overall)\b",
        query,
        flags=re.IGNORECASE,
    )
    scoped_target = re.search(
        r"\b(?:section|chapter|page|pages|paragraph|part|about|regarding)\b",
        query,
        flags=re.IGNORECASE,
    )
    if summary_intent and broad_target and not scoped_target:
        raise GuardrailViolationError(
            "Broad document or corpus summaries are not supported. "
            "Ask a focused question or specify a section, page, or topic."
        )


class EnterpriseGroundingGuard(BaseNodePostprocessor):
    """Reject queries whose best Cohere reranker score is below the threshold."""
    threshold: float

    def __init__(self, threshold: float):
        super().__init__(threshold=threshold)

    @classmethod
    def class_name(cls) -> str:
        return "EnterpriseGroundingGuard"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes:
            raise GuardrailViolationError(
                "No document passages were returned by the reranker. "
                "Check the Qdrant index and try a question related to the PDFs."
            )

        highest_score = nodes[0].score
        if highest_score is None:
            raise GuardrailViolationError(
                "The reranker returned no relevance score, so grounding could not be verified."
            )

        if highest_score < self.threshold:
            raise GuardrailViolationError(
                f"Grounding check failed: Cohere's top relevance score "
                f"({highest_score:.3f}) is below the configured threshold "
                f"({self.threshold:.3f}). Try a more specific question or lower "
                "GROUNDING_THRESHOLD if this is a relevant passage."
            )

        return nodes


class GuardrailViolationError(Exception):
    """Raised when a query or response violates corporate guardrails."""
    pass