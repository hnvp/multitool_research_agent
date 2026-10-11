from __future__ import annotations
from langchain.tools import tool
from tavily import TavilyClient
from config import settings
from trace import run_tool_safely


def _search_web(
    query: str,
) -> str:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured in .env")

    client = TavilyClient(api_key=settings.tavily_api_key)

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    lines: list[str] = []

    if response.get("answer"):
        lines.append(f"Tavily answer: " f"{response['answer']}")

    for i, item in enumerate(
        response.get("results", []),
        start=1,
    ):
        title = item.get(
            "title",
            "Untitled",
        )

        url = item.get(
            "url",
            "",
        )

        content = (
            item.get(
                "content",
                "",
            )
            .strip()
            .replace("\n", " ")
        )

        lines.append(f"[{i}] {title}\n" f"URL: {url}\n" f"Snippet: {content[:1200]}")

    if not lines:
        return "No web search results found."

    return "\n\n".join(lines)


@tool
def web_search(
    query: str,
) -> str:
    """Search the public web for current or external information.

    Use this for recent facts, external sources, news,
    public information, or topics not expected to exist
    in the private RAG knowledge base.
    """
    return run_tool_safely("web_search", _search_web, query=query)
