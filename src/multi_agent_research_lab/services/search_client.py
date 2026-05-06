"""Search client abstraction for ResearcherAgent."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from typing import Optional

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from tenacity import Retrying, stop_after_attempt, wait_exponential


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self):
        settings = get_settings()
        self.enabled = bool(settings.tavily_api_key)
        self.client = None
        if self.enabled:
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=settings.tavily_api_key)

    def _search_once(self, query: str, max_results: int) -> dict:
        if self.client is None:
            return {"results": []}
        return self.client.search(query=query, max_results=max_results)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        settings = get_settings()
        retrying = Retrying(
            stop=stop_after_attempt(settings.search_max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )

        response: Optional[dict] = None
        for attempt in retrying:
            with attempt:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(self._search_once, query, max_results)
                    try:
                        response = future.result(timeout=settings.request_timeout_seconds)
                    except FuturesTimeoutError as exc:
                        raise TimeoutError("search_timeout") from exc

        results = []
        for result in (response or {}).get("results", []):
            results.append(SourceDocument(
                title=result.get("title", ""),
                url=result.get("url"),
                snippet=result.get("content", ""),
                metadata={"score": result.get("score", 0.0)}
            ))

        return results
