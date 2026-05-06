"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import os
from typing import Any, Literal, Optional

from tenacity import Retrying, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.openai_pricing import usd_per_1k_tokens


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        provider: Optional[Literal["ollama", "openai"]] = None,
    ):
        settings = get_settings()
        self.temperature = temperature
        self.provider = provider or self._detect_provider()
        if model is not None:
            self.model = model
        elif self.provider == "ollama":
            if not settings.ollama_model:
                raise RuntimeError("OLLAMA_MODEL is required when using LLM_PROVIDER=ollama")
            self.model = settings.ollama_model
        else:
            self.model = settings.openai_model

    def _detect_provider(self) -> Literal["ollama", "openai"]:
        settings = get_settings()
        if settings.llm_provider in {"ollama", "openai"}:
            return settings.llm_provider  # type: ignore[return-value]

        if settings.openai_api_key:
            try:
                import openai  # noqa: F401
            except Exception:
                pass
            else:
                return "openai"

        try:
            import openai  # noqa: F401
        except Exception:
            pass
        else:
            return "ollama"

        try:
            import ollama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "No LLM provider available. Install extra '[llm]' and set env vars."
            ) from exc
        else:
            return "ollama"

    def _langsmith_enabled(self) -> bool:
        value = os.getenv("LANGSMITH_TRACING", "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _run_with_timeout(self, fn: Any, timeout_seconds: int) -> Any:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                raise TimeoutError("llm_timeout") from exc

    def _openai_client(self, api_key: str, base_url: Optional[str] = None) -> Any:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        if not self._langsmith_enabled():
            return client

        try:
            from langsmith.wrappers import wrap_openai
        except Exception:
            return client

        return wrap_openai(client)

    def _chat_ollama_once(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        settings = get_settings()
        try:
            client = self._openai_client(api_key="ollama", base_url=settings.ollama_base_url)
        except Exception:
            client = None

        if client is not None:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                timeout=settings.request_timeout_seconds,
            )
            choice = resp.choices[0]
            content = choice.message.content if choice.message and choice.message.content else ""
            prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
            completion_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
            return {
                "message": {"content": content},
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens,
            }

        import ollama

        if self._langsmith_enabled():
            try:
                from langsmith import trace as langsmith_trace
            except Exception:
                return ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": self.temperature},
                )
            else:
                with langsmith_trace("llm.ollama.chat", run_type="llm") as run:
                    run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
                    if run_id is not None:
                        pass
                    response = ollama.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={"temperature": self.temperature},
                    )
                    try:
                        run.end(outputs={"response": response})
                    except Exception:
                        pass
                    return response

        return ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": self.temperature},
        )

    def _chat_openai_once(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        client = self._openai_client(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            timeout=settings.request_timeout_seconds,
        )

        choice = resp.choices[0]
        content = choice.message.content if choice.message and choice.message.content else ""
        prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        completion_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)

        return {
            "message": {"content": content},
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        settings = get_settings()
        retrying = Retrying(
            stop=stop_after_attempt(settings.llm_max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )

        response: Optional[dict[str, Any]] = None
        for attempt in retrying:
            with attempt:
                if self.provider == "openai":
                    response = self._chat_openai_once(system_prompt, user_prompt)
                else:
                    response = self._run_with_timeout(
                        lambda: self._chat_ollama_once(system_prompt, user_prompt),
                        settings.request_timeout_seconds,
                    )

        usage_prompt = response.get("prompt_eval_count")
        usage_completion = response.get("eval_count")
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        input_tokens = usage_prompt if isinstance(usage_prompt, int) else None
        output_tokens = usage_completion if isinstance(usage_completion, int) else None
        cost_usd = None
        if (
            input_tokens is not None
            and output_tokens is not None
            and settings.input_token_cost_usd_per_1k is not None
            and settings.output_token_cost_usd_per_1k is not None
        ):
            cost_usd = (input_tokens / 1000.0) * settings.input_token_cost_usd_per_1k + (
                output_tokens / 1000.0
            ) * settings.output_token_cost_usd_per_1k
        elif input_tokens is not None and output_tokens is not None and self.provider == "openai":
            pricing = usd_per_1k_tokens(self.model)
            if pricing is not None:
                input_per_1k, output_per_1k = pricing
                cost_usd = (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k
        return LLMResponse(
            content=content if isinstance(content, str) else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
