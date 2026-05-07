#!/usr/bin/env python3
"""Prompt-only autoresearch for reducing memory-prompt token collapse.

This runner intentionally does not modify the v21 parser. It starts from the
cross-model parse-stable memory prompt and searches for minimal neutral prompt
edits that keep parsing intact while reducing final-state unilateral collapse.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_TEST_DIR = Path(__file__).resolve().parent
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_test"
RUN_BATCH = STREAMLIT_DIR / "run_batch_png.py"
START_PROMPT = MEMORY_TEST_DIR / "prompt_templates_memory_test.yaml"
BIAS_DIR = MEMORY_TEST_DIR / "autoresearch_memory_bias_reduction"
HINT_PATH = BIAS_DIR / "INITIAL_HINT.md"
DEFAULT_BATCH_ROOT = Path("/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000")

@dataclass(frozen=True)
class ModelEndpoint:
    tag: str
    model: str
    base_url: str


MODELS = {
    "llama8b": ModelEndpoint("llama8b", "meta-llama/Llama-3.1-8B-Instruct", "http://127.0.0.1:30000"),
    "gemma4b": ModelEndpoint("gemma4b", "google/gemma-3-4b-it", "http://127.0.0.1:30001"),
    "qwen4b_no_think": ModelEndpoint("qwen4b_no_think", "Qwen/Qwen3-4B", "http://127.0.0.1:30002"),
}


@dataclass
class CaseSummary:
    seed: int
    memory_w: int
    complete: bool
    failed: bool
    stop_reason: str | None
    consensus_round: int | None
    consensus_val: int | None
    counts: dict[str, int]
    entropy: float
    unilateral: bool
    hash_same: bool | None = None


@dataclass
class MemoryEval:
    model_tag: str
    expected: int
    completed: int
    failed: int
    returncode: int
    output_dir: Path
    cases: list[CaseSummary]
    stdout_tail: str

    @property
    def parse_ok(self) -> bool:
        return self.completed == self.expected and self.failed == 0 and self.returncode == 0

    @property
    def non_unilateral(self) -> bool:
        return self.parse_ok and all(not c.unilateral for c in self.cases)

    @property
    def score(self) -> float:
        if not self.cases:
            return 0.0
        parse_bonus = self.completed / max(1, self.expected)
        entropy = sum(c.entropy for c in self.cases) / len(self.cases)
        unilateral_penalty = sum(1 for c in self.cases if c.unilateral)
        return parse_bonus + entropy - unilateral_penalty


@dataclass
class ExtractEval:
    parse_ok: int
    total: int
    token_counts: dict[str, int]
    output_path: Path

    @property
    def passed(self) -> bool:
        return self.parse_ok == self.total and len([v for v in self.token_counts.values() if v > 0]) >= 2


FORBIDDEN = [
    r"majority\s+rule",
    r"count[- ]?first",
    r"tie[- ]?break",
    r"\bif\s+.*\bcount",
    r"\bchoose\s+the\s+majority",
    r"\bchoose\s+.*\bmost\s+frequent",
    r"\bprefer\s+\[?[kz]\]?",
    r"\bmemory\s+rule",
    r"\bposition\s+rule",
    r"\bcurrent[- ]opinion\s+rule",
    r"\bdefault\s+is\s+\[?[kz]\]?",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_prompt(path: Path) -> dict[str, str]:
    data = yaml.safe_load(read_text(path))
    prompt = data["v21_zero_shot_cot"]
    return {"system": str(prompt["system"]), "user": str(prompt["user"])}


def dump_prompt_yaml(system: str, user: str, out_path: Path) -> None:
    data = {"v21_zero_shot_cot": {"system": system.rstrip() + "\n", "user": user.rstrip() + "\n"}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def extract_yaml_block(text: str) -> dict[str, str] | None:
    fenced = re.search(r"```(?:yaml|yml)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    raw = fenced.group(1) if fenced else text
    try:
        data = yaml.safe_load(raw)
    except Exception:
        return None
    if not isinstance(data, dict) or "v21_zero_shot_cot" not in data:
        return None
    prompt = data["v21_zero_shot_cot"]
    if not isinstance(prompt, dict):
        return None
    system = prompt.get("system")
    user = prompt.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        return None
    return {"system": system, "user": user}


def validate_prompt(system: str, user: str) -> list[str]:
    errors: list[str] = []
    joined = f"{system}\n{user}"
    if "{neighbors_section}" not in user:
        errors.append("missing {neighbors_section}")
    if "MEMORY" not in user or "CURRENT ROUND" not in user:
        errors.append("candidate must preserve MEMORY and CURRENT ROUND references")
    if "k" not in joined or "z" not in joined:
        errors.append("candidate no longer names both symbols")
    if re.search(r"\[[^\]\n]*[kz][^\]\n]*\]", user, flags=re.IGNORECASE):
        errors.append("candidate should not place literal k/z inside bracket examples beyond final output wording")
    for pattern in FORBIDDEN:
        if re.search(pattern, joined, flags=re.IGNORECASE | re.DOTALL):
            errors.append(f"forbidden decision-rule pattern: {pattern}")
    return errors


def seeded_candidates(start: dict[str, str]) -> list[dict[str, str]]:
    system_kz = start["system"]
    system_zk = """You will participate in an opinion choice activity.

You must choose exactly one of two symbols: z or k. No other symbols are valid. Use only the provided input fields."""
    system_no_order = """You will participate in an opinion choice activity.

You must choose exactly one of the two valid symbols named in this task. No other symbols are valid. Use only the provided input fields. The valid symbols are z and k."""
    variants = [
        (
            system_kz,
            start["user"],
        ),
        (
            system_kz,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Base it on this specific input instance, not on the symbol names. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_kz,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Both symbols are valid labels; choose the one that your reasoning supports for this instance. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_kz,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Think step by step briefly, then write one short reasoning sentence in plain text only. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_zk,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Base it on this specific input instance, not on the symbol names. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_zk,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Treat the two symbols symmetrically as labels. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Choose according to the concrete input instance, not according to the written order of symbol names. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. No symbol is a default answer; let the concrete input instance determine the final choice. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Describe what the given input supports before choosing. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Think step by step briefly, then write one short reasoning sentence in plain text only. Treat symbol names as arbitrary labels with no built-in preference. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_zk,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. The final choice should match that sentence, not the order in which symbols are written here. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. First decide what the provided input indicates, then express only that selected symbol as the answer. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_kz,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Base it on this specific input instance, not on the symbol names. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets. The final symbol must match the symbol supported by your reasoning sentence.""",
        ),
        (
            system_kz,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Base it on this specific input instance, not on the symbol names, and make the sentence identify the same symbol you will output. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Choose according to the concrete input instance, not according to the written order of symbol names. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets. Before sending, verify that the final symbol does not contradict the reasoning sentence.""",
        ),
        (
            system_no_order,
            """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. State the basis for one symbol, then use that same symbol in the final answer. Do not use square brackets in the reasoning sentence.

On the next line, write only the chosen symbol inside square brackets.""",
        ),
    ]
    return [{"system": system, "user": user} for system, user in variants]


def expand_seeds(seeds: str) -> list[int]:
    out: list[int] = []
    for part in str(seeds).split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return out


def binary_entropy(k: int, z: int) -> float:
    total = k + z
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in (k, z):
        if value:
            p = value / total
            entropy -= p * math.log2(p)
    return entropy


def final_counts(states_path: Path) -> dict[str, int]:
    arr = np.load(states_path)
    last = arr if arr.ndim == 1 else arr[-1]
    valid = last[~np.isnan(last)] if np.issubdtype(last.dtype, np.floating) else last
    return {"k": int((valid == 0).sum()), "z": int((valid == 1).sum())}


def analyze_memory_output(output_dir: Path, expected: int, returncode: int, stdout: str) -> MemoryEval:
    cases: list[CaseSummary] = []
    completed = 0
    failed = len(list(output_dir.glob("seed_distribution_*/memory_w_*/FAILED.txt")))
    for run_dir in sorted(output_dir.glob("seed_distribution_*/memory_w_*")):
        if not run_dir.is_dir():
            continue
        seed_match = re.search(r"seed_distribution_(\d+)", str(run_dir))
        mem_match = re.search(r"memory_w_(\d+)", str(run_dir))
        seed = int(seed_match.group(1)) if seed_match else -1
        memory_w = int(mem_match.group(1)) if mem_match else -1
        failed_case = (run_dir / "FAILED.txt").exists()
        meta_path = run_dir / "result_meta.json"
        states_path = run_dir / "states.npy"
        if not meta_path.exists() or failed_case:
            cases.append(
                CaseSummary(
                    seed=seed,
                    memory_w=memory_w,
                    complete=False,
                    failed=failed_case,
                    stop_reason=None,
                    consensus_round=None,
                    consensus_val=None,
                    counts={"k": 0, "z": 0},
                    entropy=0.0,
                    unilateral=True,
                )
            )
            continue
        completed += 1
        meta = json.loads(read_text(meta_path))
        counts = final_counts(states_path) if states_path.exists() else {"k": 0, "z": 0}
        entropy = binary_entropy(counts["k"], counts["z"])
        cases.append(
            CaseSummary(
                seed=seed,
                memory_w=memory_w,
                complete=True,
                failed=False,
                stop_reason=meta.get("stop_reason"),
                consensus_round=meta.get("consensus_round"),
                consensus_val=meta.get("consensus_val"),
                counts=counts,
                entropy=entropy,
                unilateral=counts["k"] == 0 or counts["z"] == 0,
            )
        )
    return MemoryEval(
        model_tag=output_dir.name,
        expected=expected,
        completed=completed,
        failed=failed,
        returncode=returncode,
        output_dir=output_dir,
        cases=cases,
        stdout_tail=stdout[-4000:],
    )


def run_memory_eval(
    *,
    candidate_yaml: Path,
    endpoint: ModelEndpoint,
    batch_root: Path,
    label: str,
    seeds: str,
    memory_windows: list[int],
    timeout_s: int,
) -> MemoryEval:
    output_dir = batch_root / label / endpoint.tag
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_BATCH),
        "--base-url",
        endpoint.base_url,
        "--model",
        endpoint.model,
        "--api-format",
        "sglang",
        "--temperature",
        "0.0",
        "--request-seed",
        "42",
        "--max-tokens",
        "3000",
        "--repeat-penalty",
        "1.0",
        "--prompt-variant",
        "v21_zero_shot_cot",
        "--agents",
        "30",
        "--seeds-distribution",
        seeds,
        "--memory-windows",
        *[str(w) for w in memory_windows],
        "--output-dir",
        str(output_dir),
        "--mode",
        "overwrite",
    ]
    env = os.environ.copy()
    env["PROMPT_TEMPLATE_OVERRIDE_YAML"] = str(candidate_yaml)
    env.setdefault("GLOBAL_LLM_MAX_INFLIGHT", "3")
    env.setdefault("FORCE_QWEN_NO_THINK", "auto")
    proc = subprocess.run(
        cmd,
        cwd=str(STREAMLIT_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    write_text(output_dir / "runner_stdout.log", proc.stdout)
    expected = len(expand_seeds(seeds)) * len(memory_windows)
    return analyze_memory_output(output_dir, expected, int(proc.returncode), proc.stdout)


def serialize_memory_eval(result: MemoryEval) -> dict[str, Any]:
    return {
        "model_tag": result.model_tag,
        "ok_parse": result.parse_ok,
        "non_unilateral": result.non_unilateral,
        "score": result.score,
        "expected": result.expected,
        "completed": result.completed,
        "failed": result.failed,
        "returncode": result.returncode,
        "output_dir": str(result.output_dir),
        "cases": [
            {
                "seed": c.seed,
                "memory_w": c.memory_w,
                "complete": c.complete,
                "failed": c.failed,
                "stop_reason": c.stop_reason,
                "consensus_round": c.consensus_round,
                "consensus_val": c.consensus_val,
                "counts": c.counts,
                "entropy": c.entropy,
                "unilateral": c.unilateral,
            }
            for c in result.cases
        ],
    }


def derive_extract_user(memory_user: str) -> str:
    marker = "{neighbors_section}"
    if marker not in memory_user:
        raise ValueError("candidate is missing {neighbors_section}")
    tail = memory_user.split(marker, 1)[1].strip()
    return f"Input:\n{marker}\n\n{tail}".rstrip()


def parse_v21_extract_response(raw: str, token0: str = "k", token1: str = "z") -> str | None:
    text = re.sub(r"(?is)<think>\s*</think>\s*", " ", str(raw or "")).strip()
    allowed = {token0.casefold(): token0, token1.casefold(): token1}
    non_empty = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty:
        return None
    final_line = non_empty[-1]
    bare = final_line.casefold()
    if bare in allowed:
        return allowed[bare]
    bracketed = re.fullmatch(r"\[\s*([^\[\]]+?)\s*\]", final_line, flags=re.DOTALL)
    if bracketed:
        return allowed.get(bracketed.group(1).strip().casefold())
    trailing = re.search(r"(?:^|\s)(" + "|".join(re.escape(t) for t in allowed) + r")\.?\s*$", final_line, flags=re.IGNORECASE)
    if trailing:
        return allowed.get(trailing.group(1).casefold())
    return None


def run_extract_gate(system: str, user: str, output_path: Path, timeout_s: int = 900) -> ExtractEval:
    """Small direct k/z extraction gate against the Llama SGLang endpoint."""
    from transformers import AutoTokenizer

    endpoint = MODELS["llama8b"]
    tokenizer = AutoTokenizer.from_pretrained(endpoint.model, trust_remote_code=True)
    extract_user = derive_extract_user(user)
    rows: list[dict[str, Any]] = []
    token_counts = {"k": 0, "z": 0}
    session = requests.Session()
    for value in range(128):
        bits = format(value, "07b")
        opinions = ["z" if bit == "1" else "k" for bit in bits]
        current = opinions[len(opinions) // 2]
        neighbors = "\n".join(
            [
                f"Complete Opinion List: {opinions!r}",
                "Your Position: You are the opinion in the middle",
                f"Your Current Opinion: {current}",
            ]
        )
        prompt_user = extract_user.replace("{neighbors_section}", neighbors)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt_user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 3000,
                "sampling_seed": 42,
                "repetition_penalty": 1.0,
                "stop_regex": r"\[(?:k|z)\]",
                "no_stop_trim": True,
            },
        }
        response = session.post(f"{endpoint.base_url}/generate", json=payload, timeout=timeout_s)
        response.raise_for_status()
        raw = response.json().get("text", "")
        if isinstance(raw, str) and raw.startswith(prompt):
            raw = raw[len(prompt) :]
        parsed = parse_v21_extract_response(raw)
        if parsed in token_counts:
            token_counts[parsed] += 1
        rows.append({"bits": bits, "opinions": "".join(opinions), "parsed": parsed, "raw": raw})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(output_path, "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    parse_ok = sum(1 for row in rows if row["parsed"] in {"k", "z"})
    return ExtractEval(parse_ok=parse_ok, total=len(rows), token_counts=token_counts, output_path=output_path)


def serialize_extract_eval(result: ExtractEval) -> dict[str, Any]:
    return {
        "parse_ok": result.parse_ok,
        "total": result.total,
        "passed": result.passed,
        "token_counts": result.token_counts,
        "output_path": str(result.output_path),
    }


def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.rstrip().splitlines())


def write_markdown_report(path: Path, record: dict[str, Any]) -> None:
    lines = [f"# Candidate {record['iteration']:03d}", "", f"Status: `{record['status']}`", ""]
    for key in ("llama", "gemma", "qwen", "extract"):
        if key in record:
            lines.append(f"## {key}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(record[key], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    write_text(path, "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autoresearch for memory prompt bias reduction.")
    parser.add_argument("--max-iters", type=int, default=0, help="0 means evaluate all manual candidates")
    parser.add_argument("--llama-timeout-s", type=int, default=7200)
    parser.add_argument("--cross-timeout-s", type=int, default=7200)
    parser.add_argument("--extract-timeout-s", type=int, default=900)
    args = parser.parse_args(argv)

    run_id = timestamp()
    out_root = BIAS_DIR / f"run_{run_id}"
    candidate_dir = out_root / "candidates"
    eval_dir = out_root / "evals"
    batch_root = DEFAULT_BATCH_ROOT / f"memory_bias_autoresearch_{run_id}"
    for directory in (candidate_dir, eval_dir, batch_root):
        directory.mkdir(parents=True, exist_ok=True)

    hint = read_text(HINT_PATH)
    start = load_prompt(START_PROMPT)
    shutil.copy2(START_PROMPT, out_root / "START_PROMPT.yaml")
    shutil.copy2(HINT_PATH, out_root / "INITIAL_HINT.md")
    write_text(
        out_root / "RUN_CONFIG.json",
        json.dumps(
            {
                "run_id": run_id,
                "batch_root": str(batch_root),
                "start_prompt": str(START_PROMPT),
                "criteria": "100% parse, no unilateral final states, maximize final entropy",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    print(f"[RUN] out_root={out_root}", flush=True)
    print(f"[RUN] batch_root={batch_root}", flush=True)

    candidates = seeded_candidates(start)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    max_iters = args.max_iters or len(candidates)
    for idx in range(1, max_iters + 1):
        cand_path = candidate_dir / f"candidate_{idx:03d}.yaml"
        if idx <= len(candidates):
            candidate = candidates[idx - 1]
            source = "manual"
        else:
            record = {"iteration": idx, "source": "manual", "status": "no_manual_candidate"}
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] no manual candidate", flush=True)
            continue

        dump_prompt_yaml(candidate["system"], candidate["user"], cand_path)
        static_errors = validate_prompt(candidate["system"], candidate["user"])
        if static_errors:
            record = {"iteration": idx, "source": source, "status": "static_reject", "errors": static_errors}
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] static reject: {static_errors}", flush=True)
            continue

        print(f"[ITER {idx}] Llama gate seeds=1-3 memory_w=3,5 ({source})", flush=True)
        llama = run_memory_eval(
            candidate_yaml=cand_path,
            endpoint=MODELS["llama8b"],
            batch_root=batch_root,
            label=f"candidate_{idx:03d}_llama_gate",
            seeds="1-3",
            memory_windows=[3, 5],
            timeout_s=args.llama_timeout_s,
        )
        record: dict[str, Any] = {
            "iteration": idx,
            "source": source,
            "candidate_yaml": str(cand_path),
            "llama": serialize_memory_eval(llama),
        }
        if best is None or llama.score > float(best.get("llama_score", -999)):
            best = {
                "iteration": idx,
                "candidate_yaml": str(cand_path),
                "llama_score": llama.score,
                "llama": serialize_memory_eval(llama),
            }
            shutil.copy2(cand_path, out_root / "BEST_SO_FAR.yaml")
            write_text(out_root / "BEST_SO_FAR.json", json.dumps(best, ensure_ascii=False, indent=2))

        if not llama.parse_ok:
            record["status"] = "llama_parse_failed"
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] reject: Llama parse failed", flush=True)
            continue
        if not llama.non_unilateral:
            record["status"] = "llama_unilateral"
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] reject: Llama unilateral score={llama.score:.4f}", flush=True)
            continue

        print(f"[ITER {idx}] Llama passed; validating Gemma and Qwen", flush=True)
        gemma = run_memory_eval(
            candidate_yaml=cand_path,
            endpoint=MODELS["gemma4b"],
            batch_root=batch_root,
            label=f"candidate_{idx:03d}_gemma_gate",
            seeds="1-3",
            memory_windows=[3, 5],
            timeout_s=args.cross_timeout_s,
        )
        qwen = run_memory_eval(
            candidate_yaml=cand_path,
            endpoint=MODELS["qwen4b_no_think"],
            batch_root=batch_root,
            label=f"candidate_{idx:03d}_qwen_gate",
            seeds="1-3",
            memory_windows=[3, 5],
            timeout_s=args.cross_timeout_s,
        )
        record["gemma"] = serialize_memory_eval(gemma)
        record["qwen"] = serialize_memory_eval(qwen)
        if not (gemma.non_unilateral and qwen.non_unilateral):
            record["status"] = "cross_model_failed"
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] reject: cross-model failed", flush=True)
            continue

        print(f"[ITER {idx}] cross-model passed; running extraction regression gate", flush=True)
        extract = run_extract_gate(
            candidate["system"],
            candidate["user"],
            out_root / "extract_gate" / f"candidate_{idx:03d}_llama_v21_kz.jsonl",
            timeout_s=args.extract_timeout_s,
        )
        record["extract"] = serialize_extract_eval(extract)
        if not extract.passed:
            record["status"] = "extract_gate_failed"
            history.append(record)
            write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
            print(f"[ITER {idx}] reject: extraction gate failed", flush=True)
            continue

        record["status"] = "keeper"
        history.append(record)
        shutil.copy2(cand_path, out_root / "KEEPER_PROMPT_MEMORY_BIAS_REDUCED.yaml")
        write_text(out_root / "KEEPER_SUMMARY.json", json.dumps(record, ensure_ascii=False, indent=2))
        write_markdown_report(eval_dir / f"candidate_{idx:03d}.md", record)
        print(f"[ITER {idx}] KEEPER: {out_root / 'KEEPER_PROMPT_MEMORY_BIAS_REDUCED.yaml'}", flush=True)
        return 0

    no_keeper = {"status": "no_keeper", "best": best, "history_tail": history[-10:]}
    write_text(out_root / "NO_KEEPER_MEMORY_BIAS_REDUCTION.md", "# No keeper\n\n```json\n" + json.dumps(no_keeper, ensure_ascii=False, indent=2) + "\n```\n")
    print(f"[DONE] no keeper. best={best}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
