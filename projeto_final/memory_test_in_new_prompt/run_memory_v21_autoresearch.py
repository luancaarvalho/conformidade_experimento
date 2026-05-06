#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_TEST_DIR = Path(__file__).resolve().parent
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_test"
RUN_BATCH = STREAMLIT_DIR / "run_batch_png.py"
START_PROMPT = MEMORY_TEST_DIR / "prompt_templates_memory_test.yaml"
HINT_PATH = MEMORY_TEST_DIR / "autoresearch_memory_v21" / "initial_hint.md"
DEFAULT_BATCH_ROOT = Path("/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000")

MINIMAX_URL = os.getenv("MINIMAX_BASE_URL", "http://172.18.254.18:1234/v1/chat/completions")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "minimax-m2.7@8bit")


@dataclass
class EvalResult:
    label: str
    expected: int
    completed: int
    failed: int
    returncode: int
    output_dir: Path
    failures: list[dict[str, Any]]
    stdout_tail: str

    @property
    def ok(self) -> bool:
        return self.completed == self.expected and self.failed == 0 and self.returncode == 0


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
]


def validate_prompt(system: str, user: str) -> list[str]:
    errors: list[str] = []
    joined = f"{system}\n{user}"
    if "{neighbors_section}" not in user:
        errors.append("missing {neighbors_section}")
    if "MEMORY" not in user:
        errors.append("candidate removed MEMORY reference")
    if "k" not in joined or "z" not in joined:
        errors.append("candidate no longer names both symbols")
    for pattern in FORBIDDEN:
        if re.search(pattern, joined, flags=re.IGNORECASE | re.DOTALL):
            errors.append(f"forbidden decision-rule pattern: {pattern}")
    return errors


def call_minimax(system_prompt: str, user_prompt: str, *, timeout_s: int = 180) -> str:
    payload = {
        "model": MINIMAX_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2600,
    }
    response = requests.post(MINIMAX_URL, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def seeded_candidates(start: dict[str, str]) -> list[dict[str, str]]:
    system = """You will participate in an opinion choice activity.

You must choose exactly one of two symbols: k or z. No other symbols are valid. Use only the provided input fields."""

    old_stable_user = """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Think step by step before choosing. Write at least 1 and at most 5 sentences explaining your reasoning in plain text only.

Then write exactly one bracketed choice token on the last line, ensuring it matches what you concluded in your reasoning. Before sending, verify that the last non-empty line is exactly one bracketed choice token and contains nothing else."""

    concise_user = """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Write one short sentence of reasoning in plain text only. Do not use square brackets in the reasoning sentence.

On the final line, write only the chosen symbol enclosed in square brackets. Before sending, verify that there is exactly one final answer line and that the last non-empty line contains only one bracketed choice token."""

    answer_marker_user = """Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

Input:
{neighbors_section}

Think step by step briefly in plain text only. Keep all reasoning before the final line and do not use square brackets in reasoning.

Final output format:
Reasoning: one short plain-text sentence.
Choice: one final line containing only the chosen symbol inside square brackets.

Before sending, delete any extra answer lines so the last non-empty line is exactly one bracketed choice token and contains nothing else."""

    return [
        {"system": system, "user": old_stable_user},
        {"system": system, "user": concise_user},
        {"system": system, "user": answer_marker_user},
        start,
    ]


def iter_completed_runs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.glob("seed_distribution_*/memory_w_*") if p.is_dir())


def collect_failures(output_dir: Path, limit: int = 6) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for failed in sorted(output_dir.glob("seed_distribution_*/memory_w_*/FAILED.txt")):
        failures.append({"path": str(failed), "message": read_text(failed).strip()[:4000]})
        if len(failures) >= limit:
            return failures
    for req in sorted(output_dir.glob("seed_distribution_*/memory_w_*/requests.jsonl")):
        try:
            lines = req.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error") or obj.get("parsed_token") is None:
                failures.append(
                    {
                        "path": str(req),
                        "round": obj.get("round"),
                        "agent": obj.get("agent"),
                        "error": obj.get("error"),
                        "parsed_token": obj.get("parsed_token"),
                        "raw_response": str(obj.get("raw_response") or obj.get("response") or "")[:3000],
                        "prompt": str(obj.get("prompt") or "")[:3000],
                    }
                )
                if len(failures) >= limit:
                    return failures
    return failures


def evaluate_candidate(
    *,
    candidate_yaml: Path,
    batch_root: Path,
    label: str,
    seeds: str,
    memory_windows: list[int],
    timeout_s: int,
) -> EvalResult:
    output_dir = batch_root / label
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUN_BATCH),
        "--base-url",
        "http://127.0.0.1:30000",
        "--model",
        "meta-llama/Llama-3.1-8B-Instruct",
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
    proc = subprocess.run(
        cmd,
        cwd=str(STREAMLIT_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    run_dirs = iter_completed_runs(output_dir)
    completed = sum(1 for p in run_dirs if (p / "result_meta.json").exists() and not (p / "FAILED.txt").exists())
    failed = len(list(output_dir.glob("seed_distribution_*/memory_w_*/FAILED.txt")))
    expected_seeds = expand_seeds(seeds)
    expected = len(expected_seeds) * len(memory_windows)
    failures = collect_failures(output_dir)
    write_text(output_dir / "runner_stdout.log", proc.stdout)
    return EvalResult(
        label=label,
        expected=expected,
        completed=completed,
        failed=failed,
        returncode=int(proc.returncode),
        output_dir=output_dir,
        failures=failures,
        stdout_tail=proc.stdout[-4000:],
    )


def expand_seeds(seeds: str) -> list[int]:
    out: list[int] = []
    for part in str(seeds).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return out


def proposal_prompt(
    *,
    hint: str,
    current: dict[str, str],
    history: list[dict[str, Any]],
) -> tuple[str, str]:
    system = """You are optimizing a prompt for a controlled LLM experiment.

Return only a YAML document with key v21_zero_shot_cot and nested string fields system and user.
Make the smallest prompt-only change that fixes parse stability.
Do not add decision rules or token preferences."""
    recent = json.dumps(history[-5:], ensure_ascii=False, indent=2)
    user = f"""Use this research hint:

{hint}

Current prompt:

```yaml
v21_zero_shot_cot:
  system: |
{indent_block(current['system'], 4)}
  user: |
{indent_block(current['user'], 4)}
```

Recent evaluation history:

```json
{recent}
```

Return a revised YAML prompt. Preserve {{neighbors_section}} in the user template. The prompt must still use the MEMORY section as prior-round context when present. Avoid duplicated output instructions."""
    return system, user


def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.rstrip().splitlines())


def summarize_result(result: EvalResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "ok": result.ok,
        "expected": result.expected,
        "completed": result.completed,
        "failed": result.failed,
        "returncode": result.returncode,
        "output_dir": str(result.output_dir),
        "failures": result.failures[:3],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prompt-only autoresearch for v21 memory parse stability.")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--timeout-target-s", type=int, default=1800)
    ap.add_argument("--timeout-smoke-s", type=int, default=3600)
    args = ap.parse_args(argv)

    run_id = timestamp()
    out_root = MEMORY_TEST_DIR / "autoresearch_memory_v21" / f"run_{run_id}"
    batch_root = DEFAULT_BATCH_ROOT / f"memory_test_in_new_prompt_autoresearch_v21_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    batch_root.mkdir(parents=True, exist_ok=True)

    hint = read_text(HINT_PATH)
    start = load_prompt(START_PROMPT)
    shutil.copy2(START_PROMPT, out_root / "start_prompt.yaml")
    write_text(out_root / "initial_hint.md", hint)
    write_text(out_root / "RUNNING.json", json.dumps({"run_id": run_id, "batch_root": str(batch_root)}, indent=2))

    history: list[dict[str, Any]] = []
    candidates: list[dict[str, str]] = seeded_candidates(start)
    best: dict[str, Any] | None = None

    print(f"[RUN] out_root={out_root}", flush=True)
    print(f"[RUN] batch_root={batch_root}", flush=True)

    for idx in range(1, args.max_iters + 1):
        cand_dir = out_root / f"candidate_{idx:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        if idx <= len(candidates):
            candidate = candidates[idx - 1]
            source = "seeded"
        else:
            system_prompt, user_prompt = proposal_prompt(hint=hint, current=start, history=history)
            raw = call_minimax(system_prompt, user_prompt)
            write_text(cand_dir / "proposal_raw.txt", raw)
            extracted = extract_yaml_block(raw)
            if extracted is None:
                record = {"iteration": idx, "source": "minimax", "status": "invalid_yaml"}
                history.append(record)
                write_text(cand_dir / "summary.json", json.dumps(record, ensure_ascii=False, indent=2))
                print(f"[ITER {idx}] invalid YAML proposal", flush=True)
                continue
            candidate = extracted
            source = "minimax"

        errors = validate_prompt(candidate["system"], candidate["user"])
        candidate_yaml = cand_dir / "candidate_prompt.yaml"
        dump_prompt_yaml(candidate["system"], candidate["user"], candidate_yaml)
        if errors:
            record = {"iteration": idx, "source": source, "status": "rejected_static", "errors": errors}
            history.append(record)
            write_text(cand_dir / "summary.json", json.dumps(record, ensure_ascii=False, indent=2))
            print(f"[ITER {idx}] static reject: {errors}", flush=True)
            continue

        print(f"[ITER {idx}] evaluating targeted seed=3 memory_w=3,5 ({source})", flush=True)
        target = evaluate_candidate(
            candidate_yaml=candidate_yaml,
            batch_root=batch_root,
            label=f"candidate_{idx:03d}_target_seed3",
            seeds="3",
            memory_windows=[3, 5],
            timeout_s=args.timeout_target_s,
        )
        record: dict[str, Any] = {
            "iteration": idx,
            "source": source,
            "target": summarize_result(target),
        }

        if best is None or (target.completed, -target.failed) > (
            int(best.get("target_completed", 0)),
            -int(best.get("target_failed", 999)),
        ):
            best = {
                "iteration": idx,
                "target_completed": target.completed,
                "target_failed": target.failed,
                "candidate_yaml": str(candidate_yaml),
            }
            shutil.copy2(candidate_yaml, out_root / "BEST_SO_FAR.yaml")

        if not target.ok:
            record["status"] = "target_failed"
            history.append(record)
            write_text(cand_dir / "summary.json", json.dumps(record, ensure_ascii=False, indent=2))
            write_text(out_root / "latest_status.json", json.dumps(record, ensure_ascii=False, indent=2))
            print(f"[ITER {idx}] target failed completed={target.completed}/{target.expected} failed={target.failed}", flush=True)
            continue

        print(f"[ITER {idx}] target passed; evaluating smoke seeds=1-3 memory_w=3,5", flush=True)
        smoke = evaluate_candidate(
            candidate_yaml=candidate_yaml,
            batch_root=batch_root,
            label=f"candidate_{idx:03d}_smoke_1_3",
            seeds="1-3",
            memory_windows=[3, 5],
            timeout_s=args.timeout_smoke_s,
        )
        record["smoke"] = summarize_result(smoke)
        if smoke.ok:
            record["status"] = "keeper"
            history.append(record)
            write_text(cand_dir / "summary.json", json.dumps(record, ensure_ascii=False, indent=2))
            write_text(out_root / "latest_status.json", json.dumps(record, ensure_ascii=False, indent=2))
            shutil.copy2(candidate_yaml, out_root / "KEEPER_PROMPT_MEMORY_V21.yaml")
            write_text(out_root / "KEEPER_SUMMARY.json", json.dumps(record, ensure_ascii=False, indent=2))
            print(f"[ITER {idx}] KEEPER found: {out_root / 'KEEPER_PROMPT_MEMORY_V21.yaml'}", flush=True)
            return 0

        record["status"] = "smoke_failed"
        history.append(record)
        write_text(cand_dir / "summary.json", json.dumps(record, ensure_ascii=False, indent=2))
        write_text(out_root / "latest_status.json", json.dumps(record, ensure_ascii=False, indent=2))
        print(f"[ITER {idx}] smoke failed completed={smoke.completed}/{smoke.expected} failed={smoke.failed}", flush=True)

    no_keeper = {"status": "no_keeper", "best": best, "history_tail": history[-10:]}
    write_text(out_root / "NO_KEEPER_MEMORY_V21.json", json.dumps(no_keeper, ensure_ascii=False, indent=2))
    print(f"[DONE] no keeper. best={best}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
