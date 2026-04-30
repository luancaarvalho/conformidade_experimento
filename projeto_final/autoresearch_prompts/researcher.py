#!/usr/bin/env python3
"""
researcher.py — Autonomous prompt researcher (autoresearch-style).

Uses a proposal model via LM Studio to propose minimal prompt edits,
tests them against Llama 8B with test_runner.py, and keeps improvements.

Usage:
    python3 researcher.py [--max-iters N]   (default: run until parse_rate=1.0)

Results logged to results.tsv in this directory.
Best prompt is always in prompt.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests as http_requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
PROMPT_FILE = THIS_DIR / "prompt.md"
RESULTS_TSV = THIS_DIR / "results.tsv"
PROGRAM_MD = THIS_DIR / "program.md"
TEST_RUNNER = THIS_DIR / "test_runner.py"

MINIMAX_MODEL = "minimax-m2.7@8bit"
LM_STUDIO_URL = "http://172.18.254.18:1234/v1/chat/completions"
MINIMAX_TEMP = 0.6
MINIMAX_MAX_TOKENS = 8192
TIMEOUT_S = 300

SUCCESS_RUNS_NEEDED = 3   # stop after N consecutive runs at parse_rate=1.0
TARGET_PARSE_RATE = 1.0
TARGET_BIAS_MAX = 1.1     # bias_score is visualization-only; not used as success criterion


# ---------------------------------------------------------------------------
# LM Studio call (proposal model)
# ---------------------------------------------------------------------------

def call_minimax(system: str, user: str) -> tuple[str, str]:
    """Returns (thinking, response). thinking may be empty string."""
    payload = {
        "model": MINIMAX_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": MINIMAX_TEMP,
        "max_tokens": MINIMAX_MAX_TOKENS,
    }
    resp = http_requests.post(LM_STUDIO_URL, json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    thinking = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    return thinking.strip(), content.strip()


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------

def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "timestamp\titeration\tparse_rate\tbias_score\tn_success\tn_fail\tkept\tdescription\n",
            encoding="utf-8",
        )


def append_result(iteration: int, parse_rate: float, n_success: int, n_fail: int,
                  kept: bool, description: str, bias_score: float = -1.0) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kept_str = "YES" if kept else "no"
    line = f"{ts}\t{iteration}\t{parse_rate:.4f}\t{bias_score:.4f}\t{n_success}\t{n_fail}\t{kept_str}\t{description}\n"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(line)


def read_best_parse_rate() -> float:
    if not RESULTS_TSV.exists():
        return 0.0
    lines = RESULTS_TSV.read_text(encoding="utf-8").strip().splitlines()
    best = 0.0
    for line in lines[1:]:   # skip header
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def read_history_summary(n_last: int = 5) -> str:
    if not RESULTS_TSV.exists():
        return "(no history yet)"
    lines = RESULTS_TSV.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[max(1, len(lines) - n_last):]
    return "\n".join(recent)


# ---------------------------------------------------------------------------
# Run test_runner.py as subprocess
# ---------------------------------------------------------------------------

_ALL_VARIANTS: bool = False       # set by --all-variants CLI arg
_TOKENS: tuple[str, str] = ("k", "z")  # used only when not --all-variants
_FAILURES_JSONL: Optional[str] = None  # used only when not --all-variants


def run_test_runner(prompt_path: Path) -> Optional[dict]:
    """Returns parsed JSON result from test_runner, or None on error."""
    try:
        if _ALL_VARIANTS:
            cmd = [sys.executable, str(TEST_RUNNER), str(prompt_path), "--all-variants"]
        else:
            cmd = [sys.executable, str(TEST_RUNNER), str(prompt_path),
                   "--tokens", _TOKENS[0], _TOKENS[1]]
            if _FAILURES_JSONL:
                cmd += ["--failures-jsonl", _FAILURES_JSONL]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=7200,
        )
        print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            print(f"[test_runner ERROR] {proc.stderr}", flush=True)
            return None
        # Last line of stdout is JSON
        lines = proc.stdout.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return None
    except Exception as e:
        print(f"[run_test_runner] exception: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Proposal-model researcher prompt
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = """\
You are an expert prompt engineer. Your job is to fix a recurring bug in a prompt
used with meta-llama-3.1-8b-instruct: the model reasons correctly but sometimes
does NOT append the required final token in square brackets at the end.

CONTEXT:
- The prompt is GENERIC — it uses {token0} and {token1} as placeholders for the actual token pair.
- The same prompt is tested against 9 different token pairs simultaneously:
  k/z, 0/1, ł/þ, p/q, a/b, α/β, yes/no, △/○, ⊕/⊖
- The fix MUST work for ALL token pairs, not just one.
- Production benchmark prompts prepend a MEMORY block with previous rounds, and that block may include unknown states shown as '?'.
- The test runner now rebuilds the current ## USER template around the real MEMORY-prefixed cases, so changes
  to MEMORY interpretation and USER wording are actually evaluated.
- The active test set is intentionally capped to the highest-priority failures so iterations stay fast.
- The fix must remain parse-robust with the MEMORY block present; do not optimize only for the no-memory prompt.
- {token0}, {token1}, {neighbors_section} are Python str.format placeholders — keep them intact.

FAILURE PATTERNS — active deterministic modes:
1) MISSING BRACKET: the model writes reasoning, states a clear conclusion in prose,
then STOPS without writing the required final bracketed token line.

2) MEMORY / NEIGHBORHOOD LOOP: with MEMORY present, the model can enter long repetitive
reasoning about position, nearby opinions, left/right neighbors, or previous rounds and then never
emit the bracket. Treat this as a formatting/termination robustness problem, not as permission to
change the experimental decision semantics.

3) LIST ECHO / ARRAY OUTPUT: the model may copy or serialize the complete opinion list instead
of producing only one final bracketed choice token. Square brackets must be reserved for the final
choice token only.

CRITICAL SEMANTIC CONSTRAINT FROM THE HUMAN EXPERIMENT OWNER:
Do NOT add any instruction that tells the participant how to choose between the two options.
The prompt must not introduce a decision policy such as majority rule, strict count-first ordering,
left/right side rules, current-opinion rules, position rules, memory rules, or tie-break rules.
Autoresearch must optimize parse robustness and termination only. It must preserve the experimental
choice semantics rather than guiding the model toward a particular logic for selecting an option.

In particular, proposals are INVALID if they say or imply any of the following:
- count the tokens and choose the larger count;
- if one count is larger, choose it immediately;
- use position/side/current opinion/memory only when counts tie;
- ignore memory, current opinion, middle position, left side, or right side when one count is larger;
- choose by overall majority, local majority, side majority, or any deterministic tie-break rule.

Research direction:
Focus on neutral structural changes only: response format, final-line enforcement, avoiding list echo,
limiting repetitive prose, and clarifying that memory/current information is context shown to the participant,
not a required decision algorithm. If you change MEMORY presentation, do it only to reduce parsing loops,
not to define a voting rule.

STRATEGY CHANGE REQUIRED — use only neutral structural wording:
A) The final sentence may contain the bracketed token, but you must describe this neutrally.
   Do NOT suggest or quote a concrete sentence pattern such as "Therefore, I choose [{token0}]."
   or any equivalent formulation.
B) REMOVE or simplify explicit section labels like "STEP 1:"/"STEP 2:" if they make the model
   stop before the bracket.
C) REMOVE labels such as "REASONING:" if they make the model treat the answer as a separate,
   optional block.

The REASONING must come BEFORE the bracket (chain-of-thought first, answer last).
Do NOT put the bracket before the reasoning.

Your proposals must:
- Keep CoT reasoning: model thinks first, bracket comes at the END (not beginning)
- NOT change the task semantics or add any decision policy
- Be as small as possible
- MUST use {token0} and {token1} placeholders — NEVER hardcode specific tokens
- MUST describe the final sentence neutrally, without giving any concrete sentence template
- MUST NOT include quoted examples such as "Therefore, I choose [...]", "I choose [...]", or similar
- MUST NOT bias toward one token or one wording pattern
- Do NOT ask the model to choose by majority, counts, side, position, current opinion, memory, or tie-break rule
- You may consider whether the MEMORY block should be interpreted more narrowly or presented differently, but start with the smallest prompt changes that can fix the loop

KNOWN BAD PATTERNS (tested, broke production — do NOT use):
- "List exactly N key factors" + "Stop after the Nth factor": model stops before bracket
- "at most N steps" / "In at most N steps": tested, produced 0.0769 parse_rate
- Any use of "Stop" that terminates before the bracket
- "DECISION: I choose [x]" line before the bracket: model stops before writing DECISION
- Adding bridging sentences / extra IMPORTANT lines after STEP 1: 3 iterations, none worked
- Any concrete example sentence containing the answer token, even with placeholders, such as
  "Therefore, I choose [{token0}]." / "Therefore, I choose [{token1}]."
- Any wording like "sentence like ..." followed by a concrete answer pattern
- Answer-first (bracket before reasoning): breaks CoT — NOT acceptable
Focus on neutral structural changes only. Avoid wording that invites the model to compare left vs right when the overall majority is not tied. Prefer minimal and punctual fixes first so we can ship quickly; only explore memory-format changes if smaller prompt changes do not resolve the live benchmark failure.

Success criterion: parse_rate == 1.00 for ALL 9 token pair variants.

Respond with:
1. A brief hypothesis (1-2 sentences) about why the current prompt fails
2. The proposed minimal change
3. The complete new ## SYSTEM and ## USER sections (ready to paste into prompt.md)

IMPORTANT: Output MUST include the full prompt in exactly this format:

```prompt
## SYSTEM
<full system prompt here — use {token0} and {token1} placeholders>

## USER
<full user prompt here — use {token0}, {token1}, {neighbors_section} placeholders>
```
"""


def ask_proposal_model_for_proposal(current_prompt: str, history: str,
                              failed_examples: list[dict],
                              initial_hint: str = "") -> Optional[str]:
    """Returns the new prompt text (between ```prompt ... ```) or None."""
    fail_text = ""
    for ex in failed_examples:
        variant = ex.get("variant", "")
        raw = ex.get("raw_response", "(unknown)")
        majority = ex.get("majority", "?")
        variant_tag = f" [{variant}]" if variant else ""
        total_len = len(raw)
        # Keep proposal context bounded; huge loop outputs cause proposal-model timeouts.
        if total_len > 2400:
            snippet = raw[:1200] + "\n...[truncated repetitive middle]...\n" + raw[-1200:]
        else:
            snippet = raw
        fail_text += (
            f"\n---{variant_tag} majority={majority} | response_length={total_len} chars\n"
            f"Model response excerpt:\n{snippet}\n"
        )

    hint_section = f"\n## Human expert hint (prioritize this above all else)\n\n{initial_hint}\n" if initial_hint else ""

    user_msg = f"""\
## Current prompt (prompt.md)

{current_prompt}

## Recent experiment history (results.tsv)

{history}

## Recent failed responses from Llama 8B (from multiple token variants)

{fail_text if fail_text else "(no failures in last run — improving)"}
{hint_section}
## Task

Propose the next minimal change to fix the missing-token bug.

Recommended first investigation:
- Do NOT overfit to a single fresh failure. The active failure set now combines the earlier missing-bracket cases with the newest contradiction-loop case; the next prompt must improve both patterns at once.
- Treat the repeated loop as a reasoning-control/termination problem, not as a reason to prescribe how the choice should be made.
- Preserve the strict last-line bracket requirement while avoiding any wording that defines a voting, majority, side, memory, position, or tie-break policy.
- The newest production failures include long contradiction loops. Fix the loop by simplifying format and termination constraints, not by imposing a decision algorithm.
- Prefer changes that reduce self-conflicting reasoning without telling the model what evidence should determine the option.
- The new production failures happen when a MEMORY block is prepended before the task. The test runner now preserves
  that MEMORY block while applying the current USER template, so proposals may change how the USER section frames
  the memory and neighborhood state.
- Several failing cases contain unknown history markers shown as '?', and the model starts inventing tie-break rules or stopping after a prose conclusion. Prefer wording that treats unknown history as context only and does not relax the final bracket requirement.
- Prefer as the first experiment a neutral two-part structure: brief reasoning first, then a separate final line containing exactly one bracketed choice token and nothing else.
- Do not include any concrete sentence template, and do not show any literal token example beyond placeholders.

MANDATORY RULES for your output prompt:
1. You MUST use {{token0}} and {{token1}} as placeholders throughout — NEVER hardcode k, z, 0, 1, or any specific token.
2. Keep {{neighbors_section}} placeholder intact in the USER section.
3. The {{token0}}, {{token1}}, {{neighbors_section}} are Python format placeholders — write them exactly as shown with double braces in this instruction but single braces in the actual prompt text.
4. Do NOT include any quoted or illustrative sentence template for the final answer.
   Wrong: "Therefore, I choose [{{token0}}]."
   Wrong: "end with a sentence like ..."
   Correct: describe the structure neutrally, e.g. say that the final sentence must contain exactly one bracketed choice token.

The proposal will be automatically rejected if {{token0}} or {{token1}} are missing from the output.
"""

    print("[proposal model] Requesting proposal...", flush=True)
    thinking, response = call_minimax(RESEARCHER_SYSTEM, user_msg)
    if thinking:
        print(f"[proposal model thinking] {thinking[:300]}...", flush=True)
    print(f"[proposal model response] {response[:400]}...", flush=True)

    # Extract ```prompt ... ``` block
    match = re.search(r"```prompt\s*\n(.*?)```", response, re.DOTALL)
    if not match:
        return None
    proposed = match.group(1).strip()

    forbidden_patterns = [
        r"strict overall majority",
        r"overall majority",
        r"global majority",
        r"majority rule",
        r"one count is larger",
        r"count[s]? (?:is|are) larger",
        r"larger count",
        r"choose (?:that|the) token",
        r"choose .* majority",
        r"tie-break",
        r"tie break",
        r"left side",
        r"right side",
        r"side information",
        r"ignore .*memory",
        r"ignore .*current opinion",
        r"ignore .*position",
    ]
    proposed_lower = proposed.lower()
    for pat in forbidden_patterns:
        if re.search(pat, proposed_lower):
            print(f"[researcher] WARNING: proposal adds forbidden decision policy ({pat}). Skipping.", flush=True)
            return None

    # Validate placeholders — if hardcoded tokens were used, skip to avoid misleading test
    if "{token0}" not in proposed or "{token1}" not in proposed:
        print("[researcher] WARNING: proposal-model output missing {token0}/{token1} placeholders. Skipping.", flush=True)
        return None

    return proposed


# ---------------------------------------------------------------------------
# Git commit helper
# ---------------------------------------------------------------------------

def git_commit(message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", str(PROMPT_FILE), str(RESULTS_TSV)],
            cwd=THIS_DIR, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=THIS_DIR, check=True, capture_output=True,
        )
        print(f"[git] committed: {message}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[git] commit failed (non-fatal): {e.stderr}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--tokens", nargs=2, metavar=("TOKEN0", "TOKEN1"), default=["k", "z"])
    parser.add_argument("--failures-jsonl", default=None,
                        help="JSONL with production failures; passed to test_runner")
    parser.add_argument("--all-variants", action="store_true",
                        help="Test against all 8 token variants simultaneously (recommended)")
    parser.add_argument("--initial-hint", default="",
                        help="Expert hint passed to proposal model on iteration 1 (and retained until a keeper is found)")
    parser.add_argument("--initial-hint-file", default="",
                        help="Path to file containing the expert hint (overrides --initial-hint)")
    args = parser.parse_args()

    global _ALL_VARIANTS, _TOKENS, _FAILURES_JSONL
    _ALL_VARIANTS = args.all_variants
    _TOKENS = tuple(args.tokens)
    _FAILURES_JSONL = args.failures_jsonl

    initial_hint = args.initial_hint
    if args.initial_hint_file:
        from pathlib import Path as _Path
        initial_hint = _Path(args.initial_hint_file).read_text(encoding="utf-8").strip()
    args.initial_hint = initial_hint

    init_results_tsv()

    # Step 0: baseline run on current prompt.md
    print("=" * 60)
    print("BASELINE: testing current prompt.md")
    print("=" * 60)
    baseline = run_test_runner(PROMPT_FILE)
    if baseline is None:
        print("ERROR: baseline test failed. Check LM Studio is running.", file=sys.stderr)
        sys.exit(1)

    best_rate = baseline["parse_rate"]
    best_bias = baseline.get("bias_score", 1.0)
    append_result(0, best_rate, baseline["n_success"], baseline["n_fail"],
                  kept=True, description="baseline v21_zero_shot_cot",
                  bias_score=best_bias)
    print(f"\nBaseline parse_rate: {best_rate:.4f}  bias_score: {best_bias:.4f}\n")

    consecutive_perfect = 0
    iteration = 0

    while iteration < args.max_iters:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} | best so far: {best_rate:.4f}")
        print(f"{'='*60}")

        # Load current best prompt and history
        current_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        history = read_history_summary()
        last_result_path = RESULTS_TSV

        # Get last failed examples from most recent run (parse results.tsv for last row)
        # For simplicity, run test_runner once on current prompt to collect failures
        # (re-use baseline on first iter, otherwise retest)
        if iteration == 1:
            failed_examples = baseline.get("failed_responses", [])
        else:
            retest = run_test_runner(PROMPT_FILE)
            if retest:
                failed_examples = retest.get("failed_responses", [])
                retest_bias = retest.get("bias_score", 1.0)
                if retest["parse_rate"] >= TARGET_PARSE_RATE:
                    consecutive_perfect += 1
                    append_result(iteration, retest["parse_rate"],
                                  retest["n_success"], retest["n_fail"],
                                  kept=True, description="retest — perfect",
                                  bias_score=retest_bias)
                    if consecutive_perfect >= SUCCESS_RUNS_NEEDED:
                        print(f"\nSUCCESS: parse_rate=1.0 for {SUCCESS_RUNS_NEEDED} consecutive runs!")
                        break
                    continue
                else:
                    consecutive_perfect = 0
            else:
                failed_examples = []

        # Ask the proposal model for a proposal
        # Pass the hint on every iteration until a keeper is found that confirms improvement
        active_hint = args.initial_hint if (args.initial_hint and iteration <= 5) else ""
        new_prompt_text = ask_proposal_model_for_proposal(current_prompt, history, failed_examples,
                                                          initial_hint=active_hint)
        if new_prompt_text is None:
            print("[researcher] Could not parse proposal-model output. Skipping iteration.", flush=True)
            append_result(iteration, -1.0, 0, 0, kept=False, description="proposal model parse error")
            time.sleep(2)
            continue

        # Write proposed prompt to temp file and test
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                         dir=THIS_DIR, delete=False,
                                         encoding="utf-8") as tf:
            tf.write(new_prompt_text)
            tmp_path = Path(tf.name)

        print(f"\n[researcher] Testing proposed prompt: {tmp_path.name}", flush=True)
        result = run_test_runner(tmp_path)
        tmp_path.unlink(missing_ok=True)

        if result is None:
            print("[researcher] Test run failed. Skipping.", flush=True)
            append_result(iteration, -1.0, 0, 0, kept=False, description="test_runner error")
            continue

        new_rate = result["parse_rate"]
        new_bias = result.get("bias_score", 1.0)
        meets_targets = new_rate >= TARGET_PARSE_RATE
        # Keep if parse_rate improves (bias is visualization-only, not a criterion)
        kept = meets_targets and new_rate > best_rate

        if kept:
            best_rate = new_rate
            best_bias = new_bias
            PROMPT_FILE.write_text(new_prompt_text, encoding="utf-8")
            print(f"\n[researcher] IMPROVEMENT: parse_rate={new_rate:.4f} — updating prompt.md", flush=True)
            git_commit(f"autoresearch iter={iteration} parse_rate={new_rate:.4f}")
        else:
            print(f"\n[researcher] No improvement (parse_rate={new_rate:.4f}, best={best_rate:.4f}). Keeping current.", flush=True)

        # Extract short description from proposal-model response for logging
        description = new_prompt_text.split("\n")[0].strip()[:80] or f"iter{iteration}"
        append_result(iteration, new_rate, result["n_success"], result["n_fail"],
                      kept=kept, description=description, bias_score=new_bias)

        if new_rate >= TARGET_PARSE_RATE:
            consecutive_perfect += 1
            if consecutive_perfect >= SUCCESS_RUNS_NEEDED:
                print(f"\nSUCCESS: parse_rate=1.0 for {SUCCESS_RUNS_NEEDED} consecutive runs!")
                break
        else:
            consecutive_perfect = 0

        time.sleep(1)

    print(f"\nDone. Best parse_rate: {best_rate:.4f}")
    print(f"Final prompt saved in: {PROMPT_FILE}")
    print(f"Full history in: {RESULTS_TSV}")


if __name__ == "__main__":
    main()
