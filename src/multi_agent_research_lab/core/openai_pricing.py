from typing import Dict, Optional, Tuple


_USD_PER_1M_TOKENS_STANDARD: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def usd_per_1k_tokens(model: str) -> Optional[Tuple[float, float]]:
    pricing = _USD_PER_1M_TOKENS_STANDARD.get(model)
    if pricing is None:
        return None
    input_per_1m, output_per_1m = pricing
    return input_per_1m / 1000.0, output_per_1m / 1000.0
