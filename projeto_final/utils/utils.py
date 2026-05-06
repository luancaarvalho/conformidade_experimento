from __future__ import annotations

import re
import time
import os
from typing import Any, Dict, Optional, Sequence

import requests

from .initial_distribution import InitialDistribution, clamp_ratio, generate_initial_distribution


def _base_url_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def responses_api_url(base_url: str) -> str:
    return f"{_base_url_root(base_url)}/v1/responses"


def extract_output_text_from_responses(data: Dict[str, Any]) -> str:
    """
    Extract text from LM Studio/OpenAI-compatible /v1/responses payload.
    """
    raw_response = ""

    if isinstance(data.get("output"), list):
        for item in data["output"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content_items = item.get("content", []) or []
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                content_type = content.get("type")
                if content_type in {"output_text", "text"}:
                    raw_response = (
                        content.get("text")
                        or content.get("content")
                        or ""
                    )
                    if raw_response:
                        break
            if raw_response:
                break

    if not raw_response and isinstance(data.get("output_text"), str):
        raw_response = data.get("output_text", "") or ""

    return str(raw_response).strip()


_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_TOKENIZER_CACHE: dict = {}
_TOKENIZER_LOCK = None  # lazy init to avoid multiprocessing issues at import time


def _get_tokenizer_cached(model_id: str):
    global _TOKENIZER_LOCK
    import threading
    if _TOKENIZER_LOCK is None:
        _TOKENIZER_LOCK = threading.Lock()
    with _TOKENIZER_LOCK:
        if model_id not in _TOKENIZER_CACHE:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER_CACHE[model_id]
_BRACKET_TOKEN_RE = re.compile(r"\[\s*([^\[\]]+?)\s*\]", re.DOTALL)


def parse_opinion_token(
    response_text: str,
    *,
    allowed_tokens: Sequence[str],
    prefer_last: bool = True,
) -> Optional[str]:
    """
    Parse bracketed opinion token from model text, optionally preferring the final token.
    Example accepted forms: [k], [ k ], [... reasoning ...] ... [z]
    """
    text = str(response_text or "").strip()
    if not text:
        return None

    allowed_norm: Dict[str, str] = {}
    for tok in allowed_tokens:
        t = str(tok or "").strip()
        if t:
            allowed_norm[t.casefold()] = t
    if not allowed_norm:
        return None

    # Remove explicit thinking blocks so parsing favors the final answer section.
    text = _THINK_BLOCK_RE.sub(" ", text)

    exact = re.fullmatch(r"\[\s*([^\[\]]+?)\s*\]", text, flags=re.DOTALL)
    if exact:
        matched = (exact.group(1) or "").strip().casefold()
        if matched in allowed_norm:
            return allowed_norm[matched]

    matches = _BRACKET_TOKEN_RE.findall(text)
    if not matches:
        return None

    iterable = reversed(matches) if prefer_last else matches
    for raw in iterable:
        matched = str(raw or "").strip().casefold()
        if matched in allowed_norm:
            return allowed_norm[matched]
    return None


def looks_like_qwen_model(model_name: str) -> bool:
    return "qwen" in str(model_name or "").strip().lower()


def should_force_no_think(model_name: str) -> bool:
    """
    Qwen models should receive /no_think so the assistant emits the direct answer
    without exposing an explicit reasoning trace block.
    """
    env = os.getenv("FORCE_QWEN_NO_THINK", "auto").strip().lower()
    if env in {"0", "false", "no"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    return looks_like_qwen_model(model_name)


def append_no_think_if_needed(user_prompt: str, model_name: str) -> str:
    text = str(user_prompt or "").rstrip()
    if not should_force_no_think(model_name):
        return text
    if "/no_think" in text:
        return text
    if not text:
        return "/no_think"
    return text + "\n\n/no_think"


def call_llm_responses(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: Optional[int] = 42,
    max_output_tokens: int = 50,
    timeout_s: int = 120,
    stream: bool = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    stop_regex: Optional[str] = None,
    session: Optional[requests.Session] = None,
    api_format: str = "responses",
) -> Dict[str, Any]:
    """
    Shared LLM call function used by interface + benchmark runner.
    api_format="responses" → /v1/responses (LM Studio)
    api_format="sglang"    → /v1/chat/completions (SGLang; applies chat template
                             identically to benchmark's tokenizer.apply_chat_template + /generate)
    """
    root = _base_url_root(base_url)
    http = session if session is not None else requests

    if api_format == "sglang":
        tok = _get_tokenizer_cached(model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        url = f"{root}/generate"
        sampling_params: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_new_tokens": int(max_output_tokens),
        }
        if seed is not None:
            sampling_params["sampling_seed"] = int(seed)
        if top_k is not None:
            sampling_params["top_k"] = int(top_k)
        if top_p is not None:
            sampling_params["top_p"] = float(top_p)
        if min_p is not None:
            sampling_params["min_p"] = float(min_p)
        if repeat_penalty is not None:
            sampling_params["repetition_penalty"] = float(repeat_penalty)
        if stop_regex is not None:
            sampling_params["stop_regex"] = stop_regex
            sampling_params["no_stop_trim"] = True
        payload: Dict[str, Any] = {"text": full_prompt, "sampling_params": sampling_params}
        t0 = time.time()
        response = http.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout_s)
        status_code = int(response.status_code)
        response.raise_for_status()
        data = response.json()
        raw_response = str(data.get("text") or "")
        if raw_response.startswith(full_prompt):
            raw_response = raw_response[len(full_prompt):]
        raw_response = raw_response.strip()
        elapsed_s = time.time() - t0
        return {
            "url": url,
            "payload": payload,
            "data": data,
            "raw_response": raw_response,
            "http_status": status_code,
            "request_elapsed_s": elapsed_s,
            "usage": None,
            "response_model": model,
        }

    url = responses_api_url(base_url)
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "stream": bool(stream),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if min_p is not None:
        payload["min_p"] = float(min_p)
    if repeat_penalty is not None:
        payload["repetition_penalty"] = float(repeat_penalty)

    t0 = time.time()
    response = http.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout_s)
    status_code = int(response.status_code)
    response.raise_for_status()
    data = response.json()
    raw_response = extract_output_text_from_responses(data if isinstance(data, dict) else {})
    elapsed_s = time.time() - t0

    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        usage = None
    response_model = data.get("model") if isinstance(data, dict) else None
    response_model = str(response_model) if response_model is not None else None

    return {
        "url": url,
        "payload": payload,
        "data": data,
        "raw_response": raw_response,
        "http_status": status_code,
        "request_elapsed_s": elapsed_s,
        "usage": usage,
        "response_model": response_model,
    }


def resolve_initial_distribution_mode(
    majority_ratio: float,
    requested_mode: Optional[str] = "auto",
) -> str:
    """
    Canonical mode resolution shared by interface + benchmark.
    Rules:
    - auto/None: half_split only when ratio == 0.5, else ratio
    - half_split with ratio != 0.5 falls back to ratio
    """
    ratio = clamp_ratio(majority_ratio)
    mode = "auto" if requested_mode is None else str(requested_mode).strip().lower()
    if mode not in {"auto", "ratio", "half_split"}:
        raise ValueError(f"invalid requested_mode={requested_mode!r}")
    if mode == "auto":
        return "half_split" if abs(ratio - 0.5) <= 1e-12 else "ratio"
    if mode == "half_split" and abs(ratio - 0.5) > 1e-12:
        return "ratio"
    return mode


def generate_initial_distribution_shared(
    *,
    n_agents: int,
    seed_distribution: int,
    majority_ratio: float = 0.51,
    requested_mode: Optional[str] = "auto",
) -> InitialDistribution:
    """
    Shared initial-distribution entrypoint for interface_v2 + benchmark runner.
    """
    resolved_mode = resolve_initial_distribution_mode(majority_ratio, requested_mode)
    if resolved_mode == "half_split" and int(n_agents) % 2 != 0:
        resolved_mode = "ratio"
    return generate_initial_distribution(
        n_agents=int(n_agents),
        seed_distribution=int(seed_distribution),
        majority_ratio=float(majority_ratio),
        mode=resolved_mode,
    )
