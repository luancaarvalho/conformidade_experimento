#!/usr/bin/env python3
"""
test_runner.py — Tests a prompt variant against meta-llama-3.1-8b-instruct.

Usage:
    python3 test_runner.py <prompt_file> --failures-jsonl PATH [--tokens TOKEN0 TOKEN1]
    python3 test_runner.py <prompt_file>   # uses hardcoded k/z fallback cases

Output (stdout, last line is JSON):
    {"parse_rate": 0.85, "n_tested": 20, "n_success": 17, "n_fail": 3,
     "failed_responses": [...]}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "projeto_final"
for candidate in (WORKSPACE_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from projeto_final.utils.utils import append_no_think_if_needed, call_llm_responses
except Exception:
    def append_no_think_if_needed(prompt: str, model_path: str) -> str:
        return prompt

    call_llm_responses = None

# Number of concurrent requests — must match server --max-running-requests 3
N_CONCURRENT = 3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 3000
TIMEOUT_S = 120

OUTPUTS_DIR = Path(__file__).parent.parent / "streamlit_test/nondeterminism_teste_sglang/outputs"
NEW_FAILURES_DIR = Path(__file__).parent / "new_failures"
NEW_FAILURES_RECENT_DIR = Path(__file__).parent / "new_failures_recent"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_test"
PROMPT_TEMPLATES_YAML = WORKSPACE_ROOT / "prompt_templates.yaml"

# Keep autoresearch fast and focused on the newest structural failures.
# The most recent benchmark failures are moved to the top of each JSONL.
MAX_CASES_PER_VARIANT = {
    "v21_zero_shot_cot": 31,
    "v21_zero_shot_cot_łþ": 8,
    "v21_zero_shot_cot_pq": 8,
    "v21_zero_shot_cot_ab": 12,
    "v21_zero_shot_cot_01": 12,
    "v21_zero_shot_cot_αβ": 8,
    "v21_zero_shot_cot_yesno": 8,
    "v21_zero_shot_cot_△○": 4,
    "v21_zero_shot_cot_⊕⊖": 4,
}

def _load_merged_failure_cases(variant: str, t0: str, t1: str,
                               limit: Optional[int] = None) -> list[tuple[dict, str]]:
    """Load active production failures only, deduped by source metadata.

    This runner intentionally ignores new_failures_recent during smoke-driven
    autoresearch so a fresh smoke failure set is the only optimization target.
    """
    seen: set[str] = set()
    merged: list[tuple[dict, str]] = []
    for d in [NEW_FAILURES_DIR]:
        path = d / f"{variant}_failures.jsonl"
        if not path.exists():
            continue
        for record, majority in load_test_cases_from_jsonl(path, t0, t1):
            key = _failure_case_key(record)
            if key not in seen:
                seen.add(key)
                merged.append((record, majority))
    if not merged:
        fallback = OUTPUTS_DIR / f"batch3_memory_w_3_{variant}" / "requests_run_1.jsonl"
        if fallback.exists():
            merged = load_test_cases_from_jsonl(fallback, t0, t1)
    if limit is not None:
        merged = merged[:limit]
    return merged


def _failure_case_key(record: dict) -> str:
    """Keep structurally distinct smoke failures even when prompts partially repeat."""
    metadata = (
        record.get("source_file") or record.get("source") or "",
        record.get("seed_distribution"),
        record.get("memory_window"),
        record.get("round"),
        record.get("agent"),
    )
    if any(v not in (None, "") for v in metadata):
        return json.dumps(metadata, ensure_ascii=False, sort_keys=False)
    return record.get("user_prompt", "") or str(record.get("complete_opinion_list"))

ALL_VARIANTS: dict[str, tuple[str, str]] = {
    "v21_zero_shot_cot":       ("k",   "z"),
    "v21_zero_shot_cot_łþ":    ("ł",   "þ"),
    "v21_zero_shot_cot_pq":    ("p",   "q"),
    "v21_zero_shot_cot_ab":    ("a",   "b"),
    "v21_zero_shot_cot_01":    ("0",   "1"),
    "v21_zero_shot_cot_αβ":    ("α",   "β"),
    "v21_zero_shot_cot_yesno": ("yes", "no"),
    "v21_zero_shot_cot_△○":    ("△",   "○"),
    "v21_zero_shot_cot_⊕⊖":    ("⊕",   "⊖"),
}

# ---------------------------------------------------------------------------
# Fallback hardcoded test cases (k/z only, used when no --failures-jsonl given)
# Format: (user_prompt_str, expected_majority)
#   expected_majority: "0"=token0 majority, "1"=token1 majority, "tie"=equal
# ---------------------------------------------------------------------------
_FALLBACK_CASES: list[tuple[str, str]] = [
    # These are minimal k/z cases that reproduce the batch-3 parse failures
    # They cover: all-z neighborhood, mixed neighborhoods with different histories
    (
        "Below you can see the complete list of opinions from all participants in your group, including your own.\n\n"
        "Complete Opinion List: ['z', 'z', 'z', 'z', 'z', 'z', 'z']\n"
        "Your Position: You are the opinion in the middle\n"
        "Your Current Opinion: z\n\n"
        "Here are the principles to follow when making your choice:\n"
        "1. You must choose an opinion to support: either 'k' or 'z'\n"
        "2. Consider the opinions of all participants when making your choice\n"
        "3. You can either keep your current opinion or change it based on the group\n\n"
        "Think step by step: briefly list key factors (e.g., counts of 'k' vs 'z'), then explain how these lead to your decision.\n\n"
        "After your reasoning, write one sentence summarizing your decision, then on the next line write ONLY [k] or [z]. Nothing else should follow.",
        "1",
    ),
    (
        "Below you can see the complete list of opinions from all participants in your group, including your own.\n\n"
        "Complete Opinion List: ['z', 'z', 'k', 'z', 'z', 'z', 'k']\n"
        "Your Position: You are the opinion in the middle\n"
        "Your Current Opinion: z\n\n"
        "Here are the principles to follow when making your choice:\n"
        "1. You must choose an opinion to support: either 'k' or 'z'\n"
        "2. Consider the opinions of all participants when making your choice\n"
        "3. You can either keep your current opinion or change it based on the group\n\n"
        "Think step by step: briefly list key factors (e.g., counts of 'k' vs 'z'), then explain how these lead to your decision.\n\n"
        "After your reasoning, write one sentence summarizing your decision, then on the next line write ONLY [k] or [z]. Nothing else should follow.",
        "1",
    ),
    (
        "Below you can see the complete list of opinions from all participants in your group, including your own.\n\n"
        "Complete Opinion List: ['k', 'k', 'k', 'k', 'k', 'k', 'k']\n"
        "Your Position: You are the opinion in the middle\n"
        "Your Current Opinion: k\n\n"
        "Here are the principles to follow when making your choice:\n"
        "1. You must choose an opinion to support: either 'k' or 'z'\n"
        "2. Consider the opinions of all participants when making your choice\n"
        "3. You can either keep your current opinion or change it based on the group\n\n"
        "Think step by step: briefly list key factors (e.g., counts of 'k' vs 'z'), then explain how these lead to your decision.\n\n"
        "After your reasoning, write one sentence summarizing your decision, then on the next line write ONLY [k] or [z]. Nothing else should follow.",
        "0",
    ),
]


# ---------------------------------------------------------------------------
# Load test cases from a JSONL file (production failures)
# ---------------------------------------------------------------------------

def load_test_cases_from_jsonl(jsonl_path: Path, t0: str, t1: str,
                               limit: Optional[int] = None) -> list[tuple[dict, str]]:
    """Load (record, expected_majority) from records where parsed_label is None."""
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
    cases = []
    for r in records:
        if r.get("parsed_label") is not None:
            continue
        col = r.get("complete_opinion_list") or []
        if col:
            c0 = col.count(t0)
            c1 = col.count(t1)
            majority = "0" if c0 > c1 else "1" if c1 > c0 else "tie"
        else:
            majority = "tie"
        cases.append((r, majority))
        if limit is not None and len(cases) >= limit:
            break
    return cases


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def parse_prompt_file(path: Path) -> tuple[str, str]:
    """Parse prompt.md and return (system_template, user_template). May contain {token0}/{token1}."""
    text = path.read_text(encoding="utf-8")
    sys_match = re.search(r"## SYSTEM\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    user_match = re.search(r"## USER\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not sys_match or not user_match:
        raise ValueError(f"prompt.md missing ## SYSTEM or ## USER section in {path}")
    return sys_match.group(1).strip(), user_match.group(1).strip()


def substitute_tokens(text: str, token0: str, token1: str) -> str:
    """Replace {token0} and {token1} placeholders with actual token strings."""
    return text.replace("{token0}", token0).replace("{token1}", token1)


def extract_memory_block(user_prompt: str) -> str:
    """Return the exact production MEMORY block, without the separator blank line."""
    if not user_prompt.startswith("=== MEMORY"):
        return ""
    # New format: memory ends at "=== CURRENT ROUND ===" separator
    for marker in ("\n\n=== CURRENT ROUND ===", "\n\nBelow you can see"):
        idx = user_prompt.find(marker)
        if idx != -1:
            return user_prompt[:idx]
    return ""


def format_benchmark_neighbors_section(record: dict) -> str:
    """Match the V21 benchmark PromptStrategy.format_data_for_prompt output."""
    full_list = list(record.get("complete_opinion_list") or [])
    current_opinion = record.get("current_opinion")
    if not full_list:
        return ""
    middle = len(full_list) // 2
    if current_opinion is None:
        current_opinion = full_list[middle]
    left = full_list[:middle]
    right = full_list[middle + 1:]
    benchmark_list = left + [current_opinion] + right
    full_list_str = f"Complete Opinion List: {benchmark_list!r}"
    middle_str = "Your Position: You are the opinion in the middle"
    current_str = f"Your Current Opinion: {current_opinion}"
    return f"{full_list_str}\n{middle_str}\n{current_str}"


def render_benchmark_user_prompt(case: object, user_template: str, token0: str, token1: str) -> str:
    """Render exactly as benchmark_batch_two_runs.py does for a production case."""
    if isinstance(case, dict):
        original_user = case.get("user_prompt", "")
        memory = extract_memory_block(original_user)
        neighbors = format_benchmark_neighbors_section(case)
        if not neighbors:
            # complete_opinion_list unavailable — extract neighbors_section directly from user_prompt
            # The production user_prompt after the CURRENT ROUND block has the template prefix
            # followed by the neighbors_section. Strip the prefix to get neighbors_section.
            template_prefix = user_template.split("{neighbors_section}")[0]
            # Remove token placeholders from prefix to match production format
            template_prefix_clean = template_prefix.replace(f"[{token0}]", "[k]").replace(f"[{token1}]", "[z]")
            # Find the start of current-round block
            current_round_marker = "\n\n=== CURRENT ROUND ===\n\n"
            idx = original_user.find(current_round_marker)
            if idx != -1:
                current_block = original_user[idx + len(current_round_marker):]
            else:
                current_block = original_user
            # Strip the template prefix to get the neighbors section
            if current_block.startswith(template_prefix_clean):
                neighbors = current_block[len(template_prefix_clean):]
            else:
                # Fallback: use the full current block as the neighbors section
                neighbors = current_block
        rendered = user_template.replace("{neighbors_section}", neighbors)
        if memory:
            rendered = memory + "\n\n=== CURRENT ROUND ===\n\n" + rendered
        return append_no_think_if_needed(rendered, LLAMA_MODEL)
    return append_no_think_if_needed(substitute_tokens(str(case), token0, token1), LLAMA_MODEL)


# ---------------------------------------------------------------------------
# Production-compatible SGLang OpenAI Responses call
# ---------------------------------------------------------------------------
SGLANG_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_SEED = 42

def call_llama(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    if call_llm_responses is None:
        raise RuntimeError("call_llm_responses import failed; cannot run production-compatible test")
    t0 = time.time()
    result = call_llm_responses(
        base_url=SGLANG_BASE_URL,
        model=LLAMA_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=TEMPERATURE,
        seed=REQUEST_SEED,
        max_output_tokens=MAX_TOKENS,
        timeout_s=TIMEOUT_S,
        api_format="responses",
    )
    raw = str(result.get("raw_response") or "").strip()
    return raw, float(result.get("request_elapsed_s") or (time.time() - t0))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_BRACKET_RE = re.compile(r"\[\s*([^\[\]\s][^\[\]]*?)\s*\]")


def parse_token(text: str, allowed: tuple[str, str]) -> Optional[str]:
    matches = _BRACKET_RE.findall(text)
    allowed_lower = {t.casefold(): t for t in allowed}
    for token in reversed(matches):
        if token.strip().casefold() in allowed_lower:
            return allowed_lower[token.strip().casefold()]
    return None


# ---------------------------------------------------------------------------
# Production smoke replay
# ---------------------------------------------------------------------------

def _should_use_production_smoke_replay(cases: list[tuple[object, str]]) -> bool:
    """Use dynamic production replay for records that carry seed/memory metadata."""
    for case, _ in cases:
        if not isinstance(case, dict):
            continue
        if case.get("seed_distribution") is not None and case.get("memory_window") is not None:
            return True
    return False


def _write_prompt_variant_to_yaml(prompt_path: Path, token0: str, token1: str) -> str:
    import yaml

    system_template, user_template = parse_prompt_file(prompt_path)
    system_prompt = substitute_tokens(system_template, token0, token1)
    user_prompt = substitute_tokens(user_template, token0, token1)
    original = PROMPT_TEMPLATES_YAML.read_text(encoding="utf-8")
    data = yaml.safe_load(original)
    if "v21_zero_shot_cot" not in data or not isinstance(data["v21_zero_shot_cot"], dict):
        raise RuntimeError(f"v21_zero_shot_cot not found in {PROMPT_TEMPLATES_YAML}")
    data["v21_zero_shot_cot"]["system"] = system_prompt
    data["v21_zero_shot_cot"]["user"] = user_prompt
    PROMPT_TEMPLATES_YAML.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )
    return original


def evaluate_production_smoke_replay(prompt_path: Path, token0: str, token1: str,
                                     test_cases: list[tuple[object, str]]) -> dict:
    """Run exact production seed/memory pairs, avoiding seed x memory cross-products."""
    combos = sorted({
        (int(case["seed_distribution"]), int(case["memory_window"]))
        for case, _ in test_cases
        if isinstance(case, dict)
        and case.get("seed_distribution") is not None
        and case.get("memory_window") is not None
    })
    if not combos:
        return {"parse_rate": 1.0, "bias_score": 0.5, "n_tested": 0,
                "n_success": 0, "n_fail": 0, "failed_responses": []}

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = STREAMLIT_DIR / "batch_outputs" / "rtx6000" / f"autoresearch_smoke_replay_{stamp}"

    print(
        f"[production-smoke] Testing {len(combos)} exact production combos via run_batch_png.py: "
        f"{combos}",
        flush=True,
    )

    original_yaml = _write_prompt_variant_to_yaml(prompt_path, token0, token1)
    proc_outputs: list[tuple[tuple[int, int], subprocess.CompletedProcess[str]]] = []
    try:
        for seed, mw in combos:
            cmd = [
                sys.executable, str(STREAMLIT_DIR / "run_batch_png.py"),
                "--base-url", SGLANG_BASE_URL,
                "--model", LLAMA_MODEL,
                "--model-pool", "",
                "--prompt-variant", "v21_zero_shot_cot",
                "--temperature", str(TEMPERATURE),
                "--max-tokens", str(MAX_TOKENS),
                "--request-seed", str(REQUEST_SEED),
                "--agents", "30",
                "--neighbors", "7",
                "--initial-majority-ratio", "0.51",
                "--initial-distribution-mode", "auto",
                "--seeds-distribution", str(seed),
                "--memory-windows", str(mw),
                "--output-dir", str(out_dir),
                "--mode", "overwrite",
            ]
            print(f"[production-smoke] Running exact combo seed={seed} memory_w={mw}", flush=True)
            proc = subprocess.run(
                cmd,
                cwd=STREAMLIT_DIR,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            proc_outputs.append(((seed, mw), proc))
            print(proc.stdout, end="", flush=True)
            if proc.stderr:
                print(proc.stderr, end="", flush=True)
            combo_dir = out_dir / f"seed_distribution_{seed:04d}" / f"memory_w_{mw}"
            if proc.returncode != 0 or (combo_dir / "FAILED.txt").exists():
                print(
                    f"[production-smoke] Fail-fast after seed={seed} memory_w={mw}; "
                    "candidate will be optimized before testing later combos.",
                    flush=True,
                )
                break
    finally:
        PROMPT_TEMPLATES_YAML.write_text(original_yaml, encoding="utf-8")

    failed_responses: list[dict] = []
    failed_combos: set[tuple[int, int]] = set()
    for (seed, mw), proc in proc_outputs:
        if proc.returncode != 0:
            failed_combos.add((seed, mw))
            failed_responses.append({
                "error": f"run_batch_png.py exited {proc.returncode}",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "seed_distribution": seed,
                "memory_window": mw,
            })

    for req_path in out_dir.glob("seed_distribution_*/memory_w_*/requests.jsonl"):
        parts = req_path.parts
        seed = int(next(p for p in parts if p.startswith("seed_distribution_")).split("_")[-1])
        mw = int(next(p for p in parts if p.startswith("memory_w_")).split("_")[-1])
        tested_combos = {combo for combo, _ in proc_outputs}
        if (seed, mw) not in tested_combos:
            continue
        for line in req_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("parsed_token") is None or record.get("error"):
                failed_combos.add((seed, mw))
                failed_responses.append({
                    "case": len(failed_responses),
                    "majority": "production-smoke",
                    "raw_response": record.get("raw_response", ""),
                    "elapsed_s": record.get("request_elapsed_s"),
                    "user_prompt": record.get("user_prompt", ""),
                    "error": record.get("error"),
                    "seed_distribution": seed,
                    "memory_window": mw,
                    "round": record.get("round"),
                    "agent": record.get("agent"),
                })

    tested_combos = {combo for combo, _ in proc_outputs}
    n_fail = len(failed_combos)
    n_success = len(tested_combos) - n_fail
    for i, item in enumerate(failed_responses, 1):
        raw = str(item.get("raw_response") or "").replace("\n", " ")[:120]
        print(
            f"  [smoke-fail {i}] seed={item.get('seed_distribution')} "
            f"mw={item.get('memory_window')} round={item.get('round')} "
            f"agent={item.get('agent')} raw={raw!r}",
            flush=True,
        )

    return {
        "parse_rate": round(n_success / len(tested_combos), 4) if tested_combos else 0.0,
        "bias_score": 0.5,
        "n_tested": len(tested_combos),
        "n_success": n_success,
        "n_fail": n_fail,
        "failed_responses": failed_responses,
    }


def _format_int_selection(values: list[int]) -> str:
    """Format run_batch_png selections as a compact single CLI argument."""
    if not values:
        return ""
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}" if len(values) > 1 else str(values[0])
    return ",".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# Bias score
# ---------------------------------------------------------------------------

def compute_bias_score(choices: list[Optional[str]], majorities: list[str], token0: str, token1: str) -> float:
    """Fraction of 'token0 expected' cases where model picks token1 instead."""
    n_token0_expected = 0
    n_token1_chosen = 0
    for choice, maj in zip(choices, majorities):
        if maj == "0" and choice is not None:
            n_token0_expected += 1
            if choice == token1:
                n_token1_chosen += 1
    if n_token0_expected == 0:
        return 0.5
    return round(n_token1_chosen / n_token0_expected, 4)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_all_variants(prompt_path: Path) -> dict:
    """Test prompt against all variants that have failure cases. Returns aggregate result."""
    total_ok = 0
    total_fail = 0
    all_failed: list[dict] = []
    per_variant: dict[str, dict] = {}

    for variant_key, (t0, t1) in ALL_VARIANTS.items():
        limit = MAX_CASES_PER_VARIANT.get(variant_key)
        cases = _load_merged_failure_cases(variant_key, t0, t1, limit=limit)
        if not cases:
            continue
        print(f"\n=== {variant_key} ({t0}/{t1}) — N={len(cases)} ===", flush=True)
        result = evaluate(prompt_path, t0, t1, cases)
        per_variant[variant_key] = result
        total_ok += result["n_success"]
        total_fail += result["n_fail"]
        # Keep up to 3 failures per variant for researcher context
        for f in result["failed_responses"][:3]:
            f["variant"] = variant_key
            all_failed.append(f)

    total = total_ok + total_fail
    if not per_variant:
        return {"parse_rate": 1.0, "bias_score": 0.5, "n_tested": 0,
                "n_success": 0, "n_fail": 0, "failed_responses": [], "per_variant": {}}

    worst_key = min(per_variant, key=lambda k: per_variant[k]["parse_rate"])
    worst_rate = per_variant[worst_key]["parse_rate"]

    print(f"\n=== AGGREGATE ===", flush=True)
    for k, r in per_variant.items():
        status = "✓" if r["parse_rate"] >= 1.0 else "✗"
        print(f"  {status} {k}: {r['parse_rate']:.4f} ({r['n_success']}/{r['n_tested']})", flush=True)
    print(f"  total: {total_ok}/{total} ok | worst: {worst_key} ({worst_rate:.4f})", flush=True)

    return {
        "parse_rate": round(total_ok / total, 4) if total > 0 else 1.0,
        "worst_parse_rate": worst_rate,
        "worst_variant": worst_key,
        "bias_score": 0.5,
        "n_tested": total,
        "n_success": total_ok,
        "n_fail": total_fail,
        "failed_responses": all_failed,
        "per_variant": {k: {"parse_rate": v["parse_rate"], "n_success": v["n_success"],
                            "n_tested": v["n_tested"]} for k, v in per_variant.items()},
    }


def evaluate(prompt_path: Path, token0: str = "k", token1: str = "z",
             test_cases: Optional[list[tuple[object, str]]] = None) -> dict:
    system_template, user_template = parse_prompt_file(prompt_path)
    system_prompt = substitute_tokens(system_template, token0, token1)
    user_template = substitute_tokens(user_template, token0, token1)
    allowed = (token0, token1)
    if test_cases is not None:
        cases = test_cases
    else:
        # Substitute tokens into fallback cases user prompts
        cases = [(substitute_tokens(up, token0, token1), maj) for up, maj in _FALLBACK_CASES]
    n_tests = len(cases)

    if n_tests == 0:
        return {"parse_rate": 1.0, "bias_score": 0.5, "n_tested": 0,
                "n_success": 0, "n_fail": 0, "failed_responses": []}

    # Submit all with N_CONCURRENT workers — matches production SGLang capacity.
    results_by_idx: dict[int, tuple[str, float]] = {}
    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        future_to_idx = {
            pool.submit(call_llama, system_prompt,
                        render_benchmark_user_prompt(case, user_template, token0, token1)): i
            for i, (case, _) in enumerate(cases)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results_by_idx[idx] = fut.result()
            except Exception as e:
                results_by_idx[idx] = (f"__ERROR__: {e}", 0.0)

    n_success = 0
    n_fail = 0
    failed_responses: list[dict] = []
    all_choices: list[Optional[str]] = []
    all_majorities: list[str] = []

    for i, (case, majority) in enumerate(cases):
        raw, elapsed = results_by_idx[i]
        all_majorities.append(majority)
        if raw.startswith("__ERROR__:"):
            n_fail += 1
            all_choices.append(None)
            failed_responses.append({"case": i, "error": raw})
            print(f"  [{i+1:02d}/{n_tests}] ERR  | {raw}", flush=True)
            continue
        token = parse_token(raw, allowed)
        all_choices.append(token)
        if token in allowed:
            n_success += 1
            status = "ok"
        else:
            n_fail += 1
            status = "fail"
            failed_responses.append({
                "case": i,
                "majority": majority,
                "raw_response": raw,
                "elapsed_s": round(elapsed, 2),
                "user_prompt": render_benchmark_user_prompt(case, user_template, token0, token1),
            })
        print(f"  [{i+1:02d}/{n_tests}] {status:4s} | maj={majority} | token={token!r} | {elapsed:.1f}s",
              flush=True)

    parse_rate = n_success / n_tests if n_tests > 0 else 0.0
    bias_score = compute_bias_score(all_choices, all_majorities, token0, token1)
    return {
        "parse_rate": round(parse_rate, 4),
        "bias_score": bias_score,
        "n_tested": n_tests,
        "n_success": n_success,
        "n_fail": n_fail,
        "failed_responses": failed_responses,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_file", nargs="?", default=None)
    ap.add_argument("--tokens", nargs=2, metavar=("TOKEN0", "TOKEN1"), default=["k", "z"])
    ap.add_argument("--failures-jsonl", default=None,
                    help="JSONL file with production requests; uses records where parsed_label is None")
    ap.add_argument("--all-variants", action="store_true",
                    help="Test against all 8 token variants simultaneously")
    args = ap.parse_args()

    prompt_file = Path(args.prompt_file) if args.prompt_file else Path(__file__).parent / "prompt.md"
    if not prompt_file.exists():
        print(f"ERROR: file not found: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    if args.all_variants:
        print(f"Testing prompt: {prompt_file} [ALL VARIANTS]", flush=True)
        result = evaluate_all_variants(prompt_file)
    else:
        token0, token1 = args.tokens[0], args.tokens[1]
        test_cases = None
        if args.failures_jsonl:
            jsonl_path = Path(args.failures_jsonl)
            if not jsonl_path.exists():
                print(f"ERROR: --failures-jsonl not found: {jsonl_path}", file=sys.stderr)
                sys.exit(1)
            test_cases = load_test_cases_from_jsonl(jsonl_path, token0, token1)
            print(f"Loaded {len(test_cases)} failure cases from {jsonl_path.name}", flush=True)

        n_tests = len(test_cases) if test_cases is not None else len(_FALLBACK_CASES)
        print(f"Testing prompt: {prompt_file}", flush=True)
        print(f"Model: {LLAMA_MODEL} | tokens=[{token0}]/[{token1}] | N={n_tests} | temp={TEMPERATURE}", flush=True)
        print("-" * 60, flush=True)
        if test_cases is not None and _should_use_production_smoke_replay(test_cases):
            result = evaluate_production_smoke_replay(prompt_file, token0, token1, test_cases)
        else:
            result = evaluate(prompt_file, token0, token1, test_cases)

    print("-" * 60, flush=True)
    print(f"parse_rate={result['parse_rate']:.4f}  "
          f"({result['n_success']}/{result['n_tested']} ok, {result['n_fail']} fail) | "
          f"bias_score={result['bias_score']:.4f}")
    print()
    print(json.dumps(result))
