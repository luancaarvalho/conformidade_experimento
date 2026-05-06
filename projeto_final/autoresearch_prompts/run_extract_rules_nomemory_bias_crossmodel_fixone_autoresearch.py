#!/usr/bin/env python3
"""Focused cross-model no-memory autoresearch for the residual V21 parse misses.

This runner is intentionally separate from earlier no-memory runners.
It starts from the best deep cross-model candidate and targets the two
remaining patterns: a bare final symbol without brackets, and spaces inside
the final bracketed token. Acceptance still requires the full protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from researcher import call_minimax


ROOT = Path("/home/ncdia/luan")
PROJECT_ROOT = ROOT / "projeto_final"
THIS_DIR = PROJECT_ROOT / "autoresearch_prompts"
OUT_ROOT = PROJECT_ROOT / "extract_rules" / "sgLang" / "autoresearch_nomemory_bias_crossmodel_fixone"
START_PROMPT = (
    THIS_DIR
    / "start_prompt_extract_rules_crossmodel_fixone_best.md"
)
HINT_FILE = THIS_DIR / "hint_extract_rules_nomemory_bias_crossmodel_fixone.md"

VARIANTS = ("v9_lista_completa_meio_kz", "v21_zero_shot_cot")

FORBIDDEN_PATTERNS = [
    r"\bmemory\b",
    r"previous\s+round",
    r"previous\s+rounds",
    r"prior[-\s]+round",
    r"current[-\s]+round",
    r"round\s+context",
    r"rounds?\b",
    r"history\b",
    r"majority\s+rule",
    r"strict\s+majority",
    r"count[-\s]+first",
    r"tie[-\s]*break",
    r"left\s+side",
    r"right\s+side",
    r"side\s+rule",
    r"position\s+rule",
    r"if .*count.*larger",
    r"choose .*majority",
    r"prefer\s+\{token0\}",
    r"prefer\s+\{token1\}",
]


@dataclass(frozen=True)
class ModelEndpoint:
    tag: str
    model_path: str
    port: int
    no_think: bool = False

    @property
    def generate_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/generate"

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"


MODELS = (
    ModelEndpoint("gemma4b", "google/gemma-3-4b-it", 30001, False),
    ModelEndpoint("llama8b", "meta-llama/Llama-3.1-8B-Instruct", 30000, False),
    ModelEndpoint("qwen4b_no_think", "Qwen/Qwen3-4B", 30002, True),
)


def seeded_candidate_texts() -> list[str]:
    """Manual continuation seeds for the two residual parse failures.

    Best previous prompt failed only on: Llama final bare symbol, and Qwen
    final bracket line with internal spaces. These seeds preserve the same
    structure and add exact three-character final-line constraints.
    """
    common_system = """You will participate in an opinion choice activity.

You must choose exactly one of two symbols: {token0} or {token1}. No other symbols are valid. Use only the provided input fields."""
    common_user = """Input:
{neighbors_section}

Make your selection."""
    v9_output = "Return exactly one bracketed choice token ([{token0}] or [{token1}]) and no other text."
    v21_outputs = [
        """Think step by step before choosing, and write your reasoning briefly in plain text only. Keep all reasoning unbracketed.

Your response has exactly two non-empty lines:
1. Reasoning sentence without square brackets.
2. Final answer token only, built as opening square bracket, chosen symbol, closing square bracket.
The final answer line must be exactly three characters long: opening square bracket, one chosen symbol, closing square bracket. Do not write the chosen symbol alone. Do not put spaces inside the brackets.""",
        """Think step by step before choosing, and write one concise reasoning sentence in plain text only. The reasoning sentence must not contain square brackets.

Then write a separate final line. The final line contains exactly three characters: opening square bracket, chosen symbol, closing square bracket. There are no spaces before, after, or inside this final line. Never output a bare symbol as the final line.""",
        """Think step by step before choosing. Use exactly two non-empty lines.
Line 1 is one short reasoning sentence in plain text and contains no square brackets.
Line 2 is the final answer token only. It has exactly three characters: opening square bracket, the selected symbol, closing square bracket.
Line 2 must not contain spaces. Line 2 must not be only the selected symbol.""",
        """Think step by step before choosing, and keep visible reasoning to one plain-text sentence with no square brackets.

After the reasoning sentence, write one final line only. Build that line as opening square bracket immediately followed by the chosen symbol immediately followed by closing square bracket. The final line contains no spaces, no words, no label, and no punctuation other than the two square brackets.""",
        """Think step by step before choosing, and write brief reasoning in plain text only. Keep all reasoning unbracketed.

The final non-empty line is the machine-readable answer. It must contain exactly opening square bracket, selected one-character symbol, closing square bracket. Internal spaces make the answer invalid. A bare symbol without brackets is invalid. End immediately after the closing bracket.""",
        """Think step by step before choosing, and write one short reasoning sentence without square brackets.

Write the answer on the next line only. The answer line must have no spaces and exactly three characters: opening square bracket, chosen symbol, closing square bracket. Do not use a label. Do not write the symbol by itself. Do not add a period.""",
    ]
    return [
        (
            f"## COMMON SYSTEM\n{common_system}\n\n"
            f"## COMMON USER\n{common_user}\n\n"
            f"## V9 OUTPUT CONTRACT\n{v9_output}\n\n"
            f"## V21 OUTPUT CONTRACT\n{v21_output}\n"
        )
        for v21_output in v21_outputs
    ]


@dataclass
class PromptSpec:
    common_system: str
    common_user: str
    v9_output: str
    v21_output: str

    def as_text(self) -> str:
        return (
            f"## COMMON SYSTEM\n{self.common_system}\n\n"
            f"## COMMON USER\n{self.common_user}\n\n"
            f"## V9 OUTPUT CONTRACT\n{self.v9_output}\n\n"
            f"## V21 OUTPUT CONTRACT\n{self.v21_output}\n"
        )

    def for_variant(self, variant: str) -> dict[str, str]:
        if variant == "v9_lista_completa_meio_kz":
            output = self.v9_output
        elif variant == "v21_zero_shot_cot":
            output = self.v21_output
        else:
            raise ValueError(f"unsupported variant: {variant}")
        return {"system": self.common_system, "user": self.common_user + "\n\n" + output}


@dataclass
class RunResult:
    model_tag: str
    candidate_id: str
    variant: str
    mapping_name: str
    token0: str
    token1: str
    parse_ok: int
    total: int
    think_ok: int
    index_counts: Counter[int]
    token_counts: Counter[str]
    rows: dict[str, int | None]
    basic_csv: Path

    @property
    def parse_complete(self) -> bool:
        return self.parse_ok == self.total

    @property
    def think_complete(self) -> bool:
        return self.think_ok == self.total

    @property
    def uses_both_indices(self) -> bool:
        return len(self.index_counts) >= 2


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def section(text: str, name: str) -> str:
    match = re.search(rf"## {re.escape(name)}\s*\n(.*?)(?=\n## |\Z)", text, re.S)
    if not match:
        raise ValueError(f"missing section: {name}")
    return match.group(1).strip()


def parse_prompt_spec(text: str) -> PromptSpec:
    spec = PromptSpec(
        common_system=section(text, "COMMON SYSTEM"),
        common_user=section(text, "COMMON USER"),
        v9_output=section(text, "V9 OUTPUT CONTRACT"),
        v21_output=section(text, "V21 OUTPUT CONTRACT"),
    )
    combined = spec.as_text()
    lower = combined.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lower):
            raise ValueError(f"forbidden no-memory/decision-policy pattern: {pattern}")
    if "{neighbors_section}" not in spec.common_user:
        raise ValueError("COMMON USER must include {neighbors_section}")
    for placeholder in ("{token0}", "{token1}"):
        if placeholder not in combined:
            raise ValueError(f"prompt must include {placeholder}")
    allowed_placeholders = {"token0", "token1", "neighbors_section"}
    unknown_placeholders = sorted(set(re.findall(r"\{([A-Za-z0-9_]+)\}", combined)) - allowed_placeholders)
    if unknown_placeholders:
        raise ValueError(f"unknown placeholders are not renderable: {unknown_placeholders}")
    bracket_literals = set(re.findall(r"\[([^\]\n]+)\]", combined))
    forbidden_brackets = sorted(item for item in bracket_literals if item not in {"{token0}", "{token1}"})
    if forbidden_brackets:
        raise ValueError(f"forbidden literal bracket placeholders/tokens: {forbidden_brackets}")
    user_without_neighbors = spec.common_user.replace("{neighbors_section}", "").lower()
    for duplicated_field in ("complete opinion list", "your position", "your current opinion"):
        if duplicated_field in user_without_neighbors:
            raise ValueError(f"COMMON USER must not duplicate rendered field: {duplicated_field}")
    v9_lower = spec.v9_output.lower()
    for phrase in ("reasoning", "explain", "think step by step", "sentence", "plain text", "two parts", "step-by-step"):
        if phrase in v9_lower:
            raise ValueError(f"V9 must stay token-only; found {phrase}")
    if "{token0}" not in spec.v9_output or "{token1}" not in spec.v9_output:
        raise ValueError("V9 output contract must explicitly name both bracketed choice placeholders")
    if "exactly" not in v9_lower:
        raise ValueError("V9 output contract must explicitly require exactly one bracketed token")
    if not any(phrase in v9_lower for phrase in ("no other text", "nothing else", "no additional text")):
        raise ValueError("V9 output contract must explicitly reject any text beyond the bracketed token")
    v21_lower = spec.v21_output.lower()
    if "think step by step" not in v21_lower:
        raise ValueError("V21 must explicitly request think step by step")
    if "reason" not in v21_lower:
        raise ValueError("V21 must request reasoning")
    if "square bracket" not in v21_lower and "bracket" not in v21_lower:
        raise ValueError("V21 must explicitly isolate the bracketed final answer")
    return spec


def fill_tokens(text: str, token0: str, token1: str) -> str:
    return text.replace("{token0}", token0).replace("{token1}", token1)


def append_no_think_if_needed(text: str, endpoint: ModelEndpoint) -> str:
    if not endpoint.no_think:
        return text
    stripped = text.rstrip()
    if "/no_think" in stripped:
        return stripped
    return stripped + "\n\n/no_think"


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], endpoint: ModelEndpoint) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if endpoint.no_think:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
        except TypeError:
            # Older tokenizer versions may not expose Qwen's hard thinking switch.
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def format_current_input(opinions: list[str]) -> str:
    current = opinions[len(opinions) // 2]
    return "\n".join(
        [
            f"Complete Opinion List: {opinions!r}",
            "Your Position: You are the opinion in the middle",
            f"Your Current Opinion: {current}",
        ]
    )


def configs(token0: str, token1: str) -> list[tuple[str, list[str]]]:
    out = []
    for value in range(2**7):
        bits = format(value, "07b")
        out.append((bits, [token1 if bit == "1" else token0 for bit in bits]))
    return out


def normalize_qwen_think(text: str) -> tuple[str, bool, str]:
    raw = text or ""
    match = re.match(r"\s*<think>(.*?)</think>\s*", raw, flags=re.S | re.I)
    if not match:
        return raw, True, ""
    think_content = match.group(1)
    if think_content.strip():
        return raw[match.end() :], False, think_content
    return raw[match.end() :], True, think_content


def normalize_response(text: str, endpoint: ModelEndpoint) -> tuple[str, bool, str]:
    if endpoint.no_think:
        return normalize_qwen_think(text)
    return text or "", True, ""


def parse_choice(text: str, variant: str, endpoint: ModelEndpoint) -> tuple[str | None, str, bool, str]:
    normalized, think_ok, think_content = normalize_response(text, endpoint)
    if not think_ok:
        return None, normalized, think_ok, think_content
    raw = normalized.strip().lower()
    if variant == "v9_lista_completa_meio_kz":
        return (raw[1] if raw in {"[k]", "[z]"} else None), normalized, think_ok, think_content

    non_empty = [line.strip().lower() for line in raw.splitlines() if line.strip()]
    if not non_empty:
        return None, normalized, think_ok, think_content
    final_line = non_empty[-1]
    if final_line in {"[k]", "[z]"}:
        return final_line[1], normalized, think_ok, think_content
    if final_line in {"k", "z"}:
        return final_line, normalized, think_ok, think_content
    spaced = re.fullmatch(r"\[\s*([kz])\s*\]", final_line)
    if spaced:
        return spaced.group(1), normalized, think_ok, think_content
    return None, normalized, think_ok, think_content


def choice_index(choice: str | None, token0: str, token1: str) -> int | None:
    if choice == token0:
        return 0
    if choice == token1:
        return 1
    return None


async def wait_health(endpoint: ModelEndpoint, timeout_s: int = 15) -> None:
    deadline = datetime.now().timestamp() + timeout_s
    while datetime.now().timestamp() < deadline:
        try:
            response = requests.get(endpoint.health_url, timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"{endpoint.tag} health check failed at {endpoint.health_url}")


async def smoke_endpoint(endpoint: ModelEndpoint) -> None:
    await wait_health(endpoint)
    tokenizer = AutoTokenizer.from_pretrained(endpoint.model_path, trust_remote_code=True)
    prompt = apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": "Choose one symbol."},
            {"role": "user", "content": "Return exactly [k]." + ("\n\n/no_think" if endpoint.no_think else "")},
        ],
        endpoint,
    )
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 64,
            "sampling_seed": 42,
            "repetition_penalty": 1.0,
            "stop_regex": r"\[(?:k|z)\]",
            "no_stop_trim": True,
        },
    }
    response = requests.post(endpoint.generate_url, json=payload, timeout=120)
    response.raise_for_status()
    print(f"[smoke] {endpoint.tag} ok", flush=True)


async def evaluate_one(
    tokenizer: Any,
    endpoint: ModelEndpoint,
    candidate_id: str,
    spec: PromptSpec,
    variant: str,
    mapping_name: str,
    token0: str,
    token1: str,
    max_inflight: int,
) -> RunResult:
    out_dir = OUT_ROOT / endpoint.tag / candidate_id / mapping_name / variant
    basic_dir = out_dir / "csv" / "basic"
    extended_dir = out_dir / "csv" / "extended"
    prompt_dir = out_dir / "prompt_logs"
    log_dir = out_dir / "logs"
    for directory in (basic_dir, extended_dir, prompt_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_safe = endpoint.model_path.replace("/", "_")
    basic_csv = basic_dir / f"dados_combinados_{variant}_n7_{mapping_name}_{candidate_id}_{model_safe}.csv"
    extended_csv = extended_dir / f"dados_estendidos_{variant}_n7_{mapping_name}_{candidate_id}_{model_safe}.csv"
    prompt_log = prompt_dir / f"prompts_log_{variant}_n7_{mapping_name}_{candidate_id}_{model_safe}.txt"
    response_log = log_dir / f"respostas_log_{variant}_n7_{mapping_name}_{candidate_id}_{model_safe}.txt"

    semaphore = asyncio.Semaphore(max_inflight)
    session = requests.Session()
    variant_prompt = spec.for_variant(variant)

    async def run_config(bits: str, opinions: list[str]) -> dict[str, Any]:
        system = fill_tokens(variant_prompt["system"], token0, token1)
        user = fill_tokens(variant_prompt["user"], token0, token1).replace(
            "{neighbors_section}",
            format_current_input(opinions),
        )
        user = append_no_think_if_needed(user, endpoint)
        full_prompt = apply_chat_template(
            tokenizer,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            endpoint,
        )

        def post() -> str:
            payload = {
                "text": full_prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 3000,
                    "sampling_seed": 42,
                    "repetition_penalty": 1.0,
                    "stop_regex": r"\[(?:k|z)\]",
                    "no_stop_trim": True,
                },
            }
            response = session.post(endpoint.generate_url, json=payload, timeout=300)
            response.raise_for_status()
            response_text = response.json().get("text", "")
            if isinstance(response_text, str) and response_text.startswith(full_prompt):
                response_text = response_text[len(full_prompt) :]
            return response_text

        async with semaphore:
            response_text = await asyncio.to_thread(post)
        choice, normalized, think_ok, think_content = parse_choice(response_text, variant, endpoint)
        idx = choice_index(choice, token0, token1)
        return {
            "configuracao_binaria": bits,
            "configuracao_letras": "".join(opinions),
            "num_ones": bits.count("1"),
            "choice_token": choice,
            "choice_index": idx,
            "think_ok": think_ok,
            "think_content": think_content,
            "prompt_input": full_prompt,
            "llm_response_raw": response_text,
            "llm_response_normalized": normalized,
        }

    records = await asyncio.gather(*(run_config(bits, opinions) for bits, opinions in configs(token0, token1)))

    with basic_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model_tag",
            "variant",
            "mapping",
            "token0",
            "token1",
            "configuracao_binaria",
            "configuracao_letras",
            "num_ones",
            "choice_token",
            "choice_index",
            "think_ok",
            "logprobs_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model_tag": endpoint.tag,
                    "variant": variant,
                    "mapping": mapping_name,
                    "token0": token0,
                    "token1": token1,
                    "configuracao_binaria": record["configuracao_binaria"],
                    "configuracao_letras": record["configuracao_letras"],
                    "num_ones": record["num_ones"],
                    "choice_token": record["choice_token"],
                    "choice_index": record["choice_index"],
                    "think_ok": record["think_ok"],
                    "logprobs_json": None,
                }
            )

    with extended_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "configuracao_binaria",
            "configuracao_letras",
            "choice_token",
            "choice_index",
            "think_ok",
            "think_content",
            "prompt_input",
            "llm_response_raw",
            "llm_response_normalized",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})

    with prompt_log.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(f"--- config={record['configuracao_binaria']} ---\n{record['prompt_input']}\n\n")

    with response_log.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                f"--- config={record['configuracao_binaria']} parsed={record['choice_token']} "
                f"index={record['choice_index']} think_ok={record['think_ok']} ---\n"
                f"RAW:\n{record['llm_response_raw']}\n\nNORMALIZED:\n{record['llm_response_normalized']}\n\n"
            )

    return RunResult(
        model_tag=endpoint.tag,
        candidate_id=candidate_id,
        variant=variant,
        mapping_name=mapping_name,
        token0=token0,
        token1=token1,
        parse_ok=sum(1 for record in records if record["choice_index"] is not None),
        total=len(records),
        think_ok=sum(1 for record in records if record["think_ok"]),
        index_counts=Counter(record["choice_index"] for record in records if record["choice_index"] is not None),
        token_counts=Counter(record["choice_token"] for record in records if record["choice_token"]),
        rows={record["configuracao_binaria"]: record["choice_index"] for record in records},
        basic_csv=basic_csv,
    )


async def evaluate_model(endpoint: ModelEndpoint, candidate_id: str, spec: PromptSpec, max_inflight: int) -> dict[str, dict[str, RunResult]]:
    tokenizer = AutoTokenizer.from_pretrained(endpoint.model_path, trust_remote_code=True)
    results: dict[str, dict[str, RunResult]] = {}
    for variant in VARIANTS:
        results[variant] = {}
        for mapping_name, token0, token1 in (("normal", "k", "z"), ("swap", "z", "k")):
            result = await evaluate_one(tokenizer, endpoint, candidate_id, spec, variant, mapping_name, token0, token1, max_inflight)
            results[variant][mapping_name] = result
            print(
                f"[eval] {endpoint.tag} {candidate_id} {variant} {mapping_name}: "
                f"parse={result.parse_ok}/{result.total} think_ok={result.think_ok}/{result.total} "
                f"index_counts={dict(result.index_counts)} token_counts={dict(result.token_counts)}",
                flush=True,
            )
    return results


def aggregate_delta(normal: RunResult, swap: RunResult) -> int:
    return abs(normal.index_counts.get(0, 0) - swap.index_counts.get(0, 0))


def model_passed(results: dict[str, dict[str, RunResult]]) -> bool:
    for variant in VARIANTS:
        normal = results[variant]["normal"]
        swap = results[variant]["swap"]
        for result in (normal, swap):
            if not (result.parse_complete and result.think_complete and result.uses_both_indices):
                return False
    return True


def summarize_model(results: dict[str, dict[str, RunResult]]) -> str:
    parts = []
    for variant in VARIANTS:
        normal = results[variant]["normal"]
        swap = results[variant]["swap"]
        parts.append(
            f"{variant}: normal_idx={dict(normal.index_counts)}, swap_idx={dict(swap.index_counts)}, "
            f"aggregate_delta={aggregate_delta(normal, swap)}/128, "
            f"parse={normal.parse_ok + swap.parse_ok}/256, "
            f"think_ok={normal.think_ok + swap.think_ok}/256"
        )
    return "; ".join(parts)


def score_gemma(results: dict[str, dict[str, RunResult]]) -> float:
    score = 0.0
    for variant in VARIANTS:
        for result in (results[variant]["normal"], results[variant]["swap"]):
            score += result.parse_ok / result.total
            score += 0.25 * (result.think_ok / result.total)
            if result.uses_both_indices:
                score += 0.25
    return score


def score_all_models(results: dict[str, dict[str, dict[str, RunResult]]]) -> float:
    score = 0.0
    for model_results in results.values():
        score += score_gemma(model_results)
    return score


def propose(current: str, history: str, hint: str) -> str | None:
    system = (
        "You are a prompt researcher for a controlled academic experiment. "
        "Design minimal, neutral prompt edits. Do not introduce memory or decision rules."
    )
    user = f"""
## Current no-memory extract-rules prompt

{current}

## Recent evaluation history

{history}

## Expert hint

{hint}

## Task

Propose one next no-memory extract-rules prompt.
The task has only the current complete opinion list, current position, and current opinion.
V9 and V21 must share the same task and input wording. The only difference is the output contract:
V9 outputs only one bracketed token; V21 uses think step by step before the final bracketed token.
The V9 output contract must explicitly say to return exactly one bracketed token and no other text.
Only these placeholders are renderable: {{neighbors_section}}, {{token0}}, and {{token1}}.
Do not add placeholders such as [position], {{current_opinion}}, {{position}}, or any other unrendered field.
Do not duplicate the fields already rendered inside {{neighbors_section}}.

Do not mention memory, previous rounds, current rounds, history, or any temporal context.
Do not add majority, count-first, tie-breaker, side, position, or current-opinion rules.

The current best prompt has only two residual V21 parse patterns:
1. Llama swap config 1010000 ended with a bare selected symbol without brackets.
2. Qwen normal config 0100110 ended with a bracketed symbol containing internal spaces.
Gemma is already complete, and V9 is already complete. Avoid large rewrites that might destabilize
those passes.

Make minimal edits that force the final answer onto a separate final line without changing the task.
Prefer an exact two-line V21 response contract: one unbracketed reasoning sentence, then one final
line containing only the bracketed token. Add precise constraints that the final line has exactly
three characters, contains no spaces inside the brackets, and is invalid if it is only the chosen
symbol without brackets. Preserve the Llama-passing keeper mechanism whenever possible: explicitly
say to build the final line as opening square bracket, chosen symbol, closing square bracket.

Return only:

```prompt
## COMMON SYSTEM
...

## COMMON USER
...

## V9 OUTPUT CONTRACT
...

## V21 OUTPUT CONTRACT
...
```
""".strip()
    last_error = None
    for attempt in range(1, 4):
        try:
            _, response = call_minimax(system, user)
        except Exception as exc:  # Keep the long run alive when the proposer times out.
            last_error = exc
            print(f"[propose] attempt={attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        match = re.search(r"```prompt\s*\n(.*?)```", response, re.S)
        return match.group(1).strip() if match else None
    if last_error is not None:
        raise RuntimeError(f"proposer failed after retries: {last_error}") from last_error
    return None


def proposal_base(start_prompt: str, best_prompt: str, best_id: str) -> str:
    # The Llama-passing keeper is the safest anchor. Use a better prompt only
    # after it has beaten that anchor on cross-model score.
    if best_id.startswith("baseline_crossmodel_"):
        return start_prompt
    return best_prompt


async def evaluate_candidate(candidate_id: str, spec: PromptSpec, max_inflight: int) -> tuple[dict[str, dict[str, dict[str, RunResult]]], bool, str]:
    all_results: dict[str, dict[str, dict[str, RunResult]]] = {}

    gemma = next(model for model in MODELS if model.tag == "gemma4b")
    gemma_results = await evaluate_model(gemma, candidate_id, spec, max_inflight)
    all_results[gemma.tag] = gemma_results
    summary = [f"{gemma.tag}: {summarize_model(gemma_results)}"]
    if not model_passed(gemma_results):
        return all_results, False, " | ".join(summary)

    for endpoint in MODELS:
        if endpoint.tag == "gemma4b":
            continue
        model_results = await evaluate_model(endpoint, candidate_id, spec, max_inflight)
        all_results[endpoint.tag] = model_results
        summary.append(f"{endpoint.tag}: {summarize_model(model_results)}")

    accepted = all(model_passed(results) for results in all_results.values())
    return all_results, accepted, " | ".join(summary)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--seeded-candidates", type=int, default=8)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    hint = HINT_FILE.read_text(encoding="utf-8")
    current_text = START_PROMPT.read_text(encoding="utf-8")
    current_spec = parse_prompt_spec(current_text)
    history: list[str] = []

    print("[startup] health/smoke checks", flush=True)
    await asyncio.gather(*(smoke_endpoint(endpoint) for endpoint in MODELS))

    baseline_id = "baseline_crossmodel_" + stamp()
    baseline_results, baseline_accepted, baseline_summary = await evaluate_candidate(baseline_id, current_spec, args.max_inflight)
    best_score = score_all_models(baseline_results)
    best_prompt = current_text
    best_id = baseline_id
    history.append(f"{baseline_id}: {baseline_summary}; crossmodel_score={best_score:.4f}")
    print(f"[baseline] {history[-1]}", flush=True)

    if baseline_accepted:
        keeper = OUT_ROOT / "KEEPER_PROMPT_NOMEMORY_BIAS_CROSSMODEL_FIXONE.md"
        keeper.write_text(best_prompt.strip() + "\n", encoding="utf-8")
        print(f"[keeper] baseline accepted path={keeper}", flush=True)
        return 0

    for iteration in range(1, args.max_iters + 1):
        print(f"\n[loop] iteration={iteration}", flush=True)
        base_prompt = proposal_base(current_text, best_prompt, best_id)
        seeds = seeded_candidate_texts()
        if iteration <= min(args.seeded_candidates, len(seeds)):
            proposed = seeds[iteration - 1]
            print(f"[loop] using seeded candidate {iteration}/{len(seeds)}", flush=True)
        else:
            try:
                proposed = propose(base_prompt, "\n".join(history[-10:]), hint)
            except RuntimeError as exc:
                history.append(f"iter {iteration}: proposer failed after retries: {exc}")
                print(f"[loop] proposer failed after retries: {exc}", flush=True)
                continue
        if not proposed:
            history.append(f"iter {iteration}: no prompt block")
            print("[loop] no prompt block", flush=True)
            continue
        try:
            spec = parse_prompt_spec(proposed)
        except ValueError as exc:
            history.append(f"iter {iteration}: rejected: {exc}")
            print(f"[loop] rejected: {exc}", flush=True)
            continue

        candidate_id = f"crossmodel_iter{iteration:02d}_{stamp()}"
        candidate_path = OUT_ROOT / "candidates" / f"{candidate_id}.md"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(proposed.strip() + "\n", encoding="utf-8")

        results, is_keeper, summary = await evaluate_candidate(candidate_id, spec, args.max_inflight)
        gemma_score = score_gemma(results["gemma4b"])
        crossmodel_score = score_all_models(results)
        history.append(f"{candidate_id}: {summary}; gemma_score={gemma_score:.4f}; crossmodel_score={crossmodel_score:.4f}")
        print(f"[score] {history[-1]}", flush=True)

        if crossmodel_score > best_score:
            best_score = crossmodel_score
            best_prompt = proposed
            best_id = candidate_id
            (OUT_ROOT / "BEST_PROMPT_NOMEMORY_BIAS_CROSSMODEL_FIXONE.md").write_text(best_prompt.strip() + "\n", encoding="utf-8")

        if is_keeper:
            keeper = OUT_ROOT / "KEEPER_PROMPT_NOMEMORY_BIAS_CROSSMODEL_FIXONE.md"
            keeper.write_text(proposed.strip() + "\n", encoding="utf-8")
            print(f"[keeper] {candidate_id} accepted path={keeper}", flush=True)
            return 0

    report = OUT_ROOT / f"NO_KEEPER_NOMEMORY_BIAS_CROSSMODEL_FIXONE_{stamp()}.txt"
    report.write_text(
        "No cross-model keeper reached parse completeness plus non-unilateral choices for Gemma, Llama, and Qwen.\n"
        f"Best={best_id}\nBest cross-model score={best_score:.4f}\n\n"
        + "\n\n".join(history),
        encoding="utf-8",
    )
    print(f"[done] no keeper; best={best_id} crossmodel_score={best_score:.4f} report={report}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
