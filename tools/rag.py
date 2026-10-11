from __future__ import annotations
from langchain.tools import tool
from pinecone import Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer
from config import settings
from trace import run_tool_safely

# Load retrieval and reranking models once
embedding_model = SentenceTransformer(
    settings.rag_embedding_model,
)

reranker = CrossEncoder(
    settings.rag_reranker_model,
)


def _get_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured in .env")

    pc = Pinecone(
        api_key=settings.pinecone_api_key,
    )

    return pc.Index(settings.pinecone_index)


# Initialize the Pinecone index once and reuse it
_INDEX = _get_index()


def _search_rag(
    query: str,
    source: str | None = None,
) -> str:
    query_embedding = embedding_model.encode(
        "query: " + query,
    )

    query_kwargs = {
        "vector": query_embedding.tolist(),
        "top_k": settings.rag_top_k,
        "include_metadata": True,
    }

    if source:
        query_kwargs["filter"] = {
            "source": {
                "$eq": source,
            }
        }

    results = _INDEX.query(**query_kwargs)

    matches = results["matches"]

    if not matches:
        return "I cannot find relevant information " "in the documents."

    candidates = []

    for match in matches:
        metadata = match["metadata"]

        candidates.append(
            {
                "match": match,
                "text": metadata.get("text", ""),
            }
        )

    reranker_inputs = [(query, candidate["text"]) for candidate in candidates]

    reranker_scores = reranker.predict(
        reranker_inputs,
    )

    ranked_candidates = sorted(
        zip(candidates, reranker_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    top_candidates = ranked_candidates[: settings.rag_top_n]

    context_parts = []

    for rank, (candidate, reranker_score) in enumerate(
        top_candidates,
        start=1,
    ):
        match = candidate["match"]

        metadata = match["metadata"]

        source_name = metadata.get(
            "source",
            "Unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown",
        )

        text = metadata.get(
            "text",
            "",
        )

        similarity_score = float(match.get("score", 0.0))

        reranker_score = float(reranker_score)

        context_parts.append(
            f"[Document {rank}]\n"
            f"Source: {source_name}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Similarity score: "
            f"{similarity_score:.4f}\n"
            f"Reranker score: "
            f"{reranker_score:.4f}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


@tool
def rag_search(
    query: str,
    source: str | None = None,
) -> str:
    """Search the private/internal knowledge base.

    Use this for facts about private project documents,
    internal systems, components, configurations, specifications,
    or behavior documented in the private knowledge base.

    Do not use this for public, current, recent, or external
    information that should be obtained from the web.

    If the user asks for multiple independent facts,
    retrieve only the private/internal facts that require RAG.
    """
    return run_tool_safely("rag_search", _search_rag, query=query, source=source)
