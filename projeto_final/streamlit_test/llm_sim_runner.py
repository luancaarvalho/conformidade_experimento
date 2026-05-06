#!/usr/bin/env python3
"""
Headless runner for the Streamlit LLM conformity simulation.

Goal: reuse the same prompt construction + memory formatting semantics as
`streamlit_test/interface/interface_llama_v1.py`, but runnable in batch mode.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

_THIS_FILE = Path(__file__).resolve()
_ROOT_CANDIDATES = [_THIS_FILE.parents[1], _THIS_FILE.parents[2]]
REPO_ROOT = _ROOT_CANDIDATES[0]
for _cand in _ROOT_CANDIDATES:
    if (_cand / "prompt_templates.yaml").exists():
        REPO_ROOT = _cand
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projeto_final.utils.utils import (
    append_no_think_if_needed,
    call_llm_responses,
    generate_initial_distribution_shared,
    parse_opinion_token,
)
from projeto_final.utils.conformity_game_prompts import (
    CONFORMITY_GAME_MODE_A,
    render_conformity_game_system_prompt,
    render_conformity_game_user_prompt,
)


def parse_llm_response_token(
    text: str,
    token0: str,
    token1: str,
    *,
    prompt_variant: str,
) -> Optional[str]:
    """
    Extract and validate the model's final opinion token.

    v9 remains machine-strict: the entire response must be exactly one
    bracketed token. v21 is reasoning-oriented, so it accepts semantically
    correct final-line variants observed in the cross-model keeper validation.
    """
    raw = str(text or "").strip()
    allowed = {str(token0).casefold(): token0, str(token1).casefold(): token1}

    if (prompt_variant or "").startswith("v9_"):
        exact = {
            f"[{str(token0).casefold()}]": token0,
            f"[{str(token1).casefold()}]": token1,
        }
        return exact.get(raw.casefold())

    if (prompt_variant or "").startswith("v21_"):
        normalized = re.sub(r"(?is)<think>\s*</think>\s*", " ", raw).strip()
        non_empty = [line.strip() for line in normalized.splitlines() if line.strip()]
        if non_empty:
            final_line = non_empty[-1]
            bare = final_line.casefold()
            if bare in allowed:
                return allowed[bare]
            bracketed = re.fullmatch(r"\[\s*([^\[\]]+?)\s*\]", final_line, flags=re.DOTALL)
            if bracketed:
                token = bracketed.group(1).strip().casefold()
                return allowed.get(token)

    return parse_opinion_token(
        raw,
        allowed_tokens=(token0, token1),
        prefer_last=True,
    )


def generate_initial_opinions(
    n_agents: int,
    majority_ratio: float,
    *,
    seed_distribution: int,
    distribution_mode: str = "auto",
) -> Tuple[np.ndarray, int]:
    dist = generate_initial_distribution_shared(
        n_agents=n_agents,
        seed_distribution=seed_distribution,
        majority_ratio=majority_ratio,
        requested_mode=distribution_mode,
    )
    return dist.opinions, int(dist.initial_majority_opinion)


def get_neighbor_indices(n_agents: int, n_neighbors: int, agent_idx: int) -> Tuple[List[int], List[int]]:
    half = n_neighbors // 2
    left = [((agent_idx - i) % n_agents) for i in range(1, half + 1)]
    right = [((agent_idx + i) % n_agents) for i in range(1, half + 1)]
    return left[::-1], right


def state_to_token_kz(val: float) -> str:
    if np.isnan(val):
        return "?"
    return "k" if int(val) == 0 else "z"


def make_stop_regex(tokens: Tuple[str, str]) -> str:
    escaped = sorted([re.escape(t) for t in tokens], key=len, reverse=True)
    return r"\[(?:" + "|".join(escaped) + r")\]"


def state_to_token(val: float, token0: str, token1: str) -> str:
    if np.isnan(val):
        return "?"
    return token0 if int(val) == 0 else token1


def token_pair_for_prompt_variant(prompt_variant: str) -> Tuple[str, str]:
    """
    Keep token mapping consistent with execucao_simultanea.* token handling.
    Only v9 variants are used by run_batch_png today, but this supports all v9 pairs.
    """
    v = (prompt_variant or "").strip()
    # v9 / v21 base variants use k/z.
    if v in ("v9_lista_completa_meio", "v9_lista_completa_meio_kz"):
        return ("k", "z")
    if v in ("v21_zero_shot_cot", "v21_zero_shot_cot_kz"):
        return ("k", "z")
    if v == "v9_lista_completa_meio_ab":
        return ("a", "b")
    if v == "v21_zero_shot_cot_ab":
        return ("a", "b")
    if v == "v9_lista_completa_meio_01":
        return ("0", "1")
    if v == "v21_zero_shot_cot_01":
        return ("0", "1")
    if v == "v9_lista_completa_meio_pq":
        return ("p", "q")
    if v == "v21_zero_shot_cot_pq":
        return ("p", "q")
    if v == "v9_lista_completa_meio_αβ":
        return ("α", "β")
    if v == "v21_zero_shot_cot_αβ":
        return ("α", "β")
    if v == "v9_lista_completa_meio_△○":
        return ("△", "○")
    if v == "v21_zero_shot_cot_△○":
        return ("△", "○")
    if v == "v9_lista_completa_meio_⊕⊖":
        return ("⊕", "⊖")
    if v == "v21_zero_shot_cot_⊕⊖":
        return ("⊕", "⊖")
    if v == "v9_lista_completa_meio_łþ":
        return ("ł", "þ")
    if v == "v21_zero_shot_cot_łþ":
        return ("ł", "þ")
    if v == "v9_lista_completa_meio_yesno":
        # Same ordering used in execucao_simultanea.py mapping for yes/no.
        return ("no", "yes")
    if v == "v21_zero_shot_cot_yesno":
        return ("no", "yes")
    # _swap variants: token order inverted in the prompt
    if v == "v21_zero_shot_cot_swap":
        return ("z", "k")
    if v == "v21_zero_shot_cot_ab_swap":
        return ("b", "a")
    if v == "v21_zero_shot_cot_pq_swap":
        return ("q", "p")
    if v == "v21_zero_shot_cot_yesno_swap":
        return ("yes", "no")
    if v == "v21_zero_shot_cot_01_swap":
        return ("1", "0")
    if v == "v21_zero_shot_cot_αβ_swap":
        return ("β", "α")
    if v == "v21_zero_shot_cot_△○_swap":
        return ("○", "△")
    if v == "v21_zero_shot_cot_⊕⊖_swap":
        return ("⊖", "⊕")
    if v == "v21_zero_shot_cot_łþ_swap":
        return ("þ", "ł")
    raise ValueError(f"Unsupported prompt_variant for token mapping: {prompt_variant!r}")


def format_memory_block_timeline(
    states: np.ndarray,
    *,
    n_agents: int,
    n_neighbors: int,
    agent_idx: int,
    current_round: int,
    memory_window: int,
    token0: str,
    token1: str,
) -> str:
    if memory_window == 0 or current_round == 0:
        return ""

    start_round = max(0, current_round - memory_window)
    end_round = current_round
    if start_round >= end_round:
        return ""

    left_indices, right_indices = get_neighbor_indices(n_agents, n_neighbors, agent_idx)
    lines: List[str] = []
    lines.append("=== MEMORY (Previous Rounds) ===")
    for r in range(start_round, end_round):
        self_token = state_to_token(states[r, agent_idx], token0, token1)
        left_tokens = [state_to_token(states[r, idx], token0, token1) for idx in left_indices]
        right_tokens = [state_to_token(states[r, idx], token0, token1) for idx in right_indices]
        lines.append(f"Round {r}:")
        lines.append(f"  You: [{self_token}]")
        lines.append(f"  Left: {left_tokens}")
        lines.append(f"  Right: {right_tokens}")
    return "\n".join(lines)


def _neighbor_index_map(n_agents: int, n_neighbors: int) -> List[Tuple[List[int], List[int]]]:
    return [get_neighbor_indices(n_agents, n_neighbors, agent_idx) for agent_idx in range(n_agents)]


def _format_memory_block_timeline_precomputed(
    states: np.ndarray,
    *,
    left_indices: List[int],
    right_indices: List[int],
    agent_idx: int,
    current_round: int,
    memory_window: int,
    token0: str,
    token1: str,
) -> str:
    if memory_window == 0 or current_round == 0:
        return ""

    start_round = max(0, current_round - memory_window)
    end_round = current_round
    if start_round >= end_round:
        return ""

    lines: List[str] = ["=== MEMORY (Previous Rounds) ==="]
    for round_idx in range(start_round, end_round):
        self_token = state_to_token(states[round_idx, agent_idx], token0, token1)
        left_tokens = [state_to_token(states[round_idx, idx], token0, token1) for idx in left_indices]
        right_tokens = [state_to_token(states[round_idx, idx], token0, token1) for idx in right_indices]
        full_list = left_tokens + [self_token] + right_tokens
        lines.append(f"Round {round_idx}: {full_list!r}  (you are in the middle)")
    return "\n".join(lines)


@dataclass(frozen=True)
class _PromptJob:
    agent_idx: int
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class _PromptJobResult:
    agent_idx: int
    selected_model: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_token: Optional[str]
    error: Optional[str]
    usage: Optional[Dict[str, Any]]
    http_status: Optional[int]
    request_elapsed_s: Optional[float]
    payload: Optional[Dict[str, Any]]
    response_model: Optional[str]


def check_consensus(states_round: np.ndarray) -> Tuple[bool, Optional[int]]:
    valid = states_round[~np.isnan(states_round)]
    if len(valid) == 0:
        return False, None
    first = int(valid[0])
    return bool(np.all(valid == first)), first


@dataclass(frozen=True)
class LlmRequestConfig:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    model_pool: Optional[Tuple[str, ...]] = None
    request_seed: Optional[int] = 42
    # Sampling knobs (LM Studio / llama.cpp compatible). If None, omit from payload.
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop_regex: Optional[str] = None
    timeout_s: int = 120
    max_attempts: int = 5
    api_format: str = "responses"


@dataclass(frozen=True)
class SimulationConfig:
    prompt_variant: str
    n_agents: int
    n_rounds: int
    n_neighbors: int
    initial_majority_ratio: float
    memory_window: int
    memory_format: str = "timeline"  # timeline|json (only timeline implemented here)
    initial_distribution_mode: str = "auto"
    seed_distribution: int = 1
    # If set > n_rounds, the simulation can auto-extend by doubling rounds until max_rounds.
    # This lets long runs continue only when the system is still changing.
    max_rounds: Optional[int] = None
    # Number of consecutive identical rounds required to stop early (stabilized).
    # If None, defaults to memory_window.
    stability_window: Optional[int] = None
    # If True, only check stability at the checkpoint "end of initial horizon" (n_rounds-1).
    # The default fast path checks stability continuously to truncate repeated states early.
    stability_check_only_at_initial_horizon: bool = False
    conformity_game: bool = False
    conformity_game_mode: str = CONFORMITY_GAME_MODE_A


def _resolve_model_pool(req_cfg: LlmRequestConfig) -> Tuple[str, ...]:
    pool: List[str] = []
    if req_cfg.model_pool:
        for model_id in req_cfg.model_pool:
            m = str(model_id).strip()
            if m:
                pool.append(m)
    if not pool:
        pool.append(str(req_cfg.model).strip())
    # Remove duplicates while preserving order.
    unique_pool = list(dict.fromkeys(pool))
    return tuple(unique_pool)


def run_simulation(
    sim_cfg: SimulationConfig,
    req_cfg: LlmRequestConfig,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if sim_cfg.n_neighbors % 2 == 0:
        raise ValueError("n_neighbors must be odd")
    if sim_cfg.n_neighbors > sim_cfg.n_agents:
        raise ValueError("n_neighbors cannot exceed n_agents")
    if sim_cfg.memory_window >= sim_cfg.n_rounds:
        raise ValueError("memory_window must be < n_rounds")
    if sim_cfg.memory_format != "timeline":
        raise ValueError("Only memory_format=timeline is supported in this runner")
    if sim_cfg.n_rounds < 2:
        raise ValueError("n_rounds must be >= 2")

    strategy = None
    if not sim_cfg.conformity_game:
        # Import prompt strategy lazily. Ensure repo root is on sys.path so this works
        # when invoked from streamlit_test/.
        from prompt_strategies import get_prompt_strategy

        yaml_override = os.getenv("PROMPT_TEMPLATE_OVERRIDE_YAML", "").strip()
        yaml_path = Path(yaml_override) if yaml_override else REPO_ROOT / "prompt_templates.yaml"
        strategy = get_prompt_strategy(sim_cfg.prompt_variant, str(yaml_path))

    token0, token1 = token_pair_for_prompt_variant(sim_cfg.prompt_variant)
    # Auto-derive stop_regex for sglang /generate to stop at the first valid token
    _stop_regex = req_cfg.stop_regex
    if req_cfg.api_format == "sglang" and _stop_regex is None:
        _stop_regex = make_stop_regex((token0, token1))
    model_pool = _resolve_model_pool(req_cfg)

    initial_rounds = int(sim_cfg.n_rounds)
    max_rounds = int(sim_cfg.max_rounds) if sim_cfg.max_rounds is not None else initial_rounds
    if max_rounds < initial_rounds:
        max_rounds = initial_rounds

    stability_window = sim_cfg.stability_window
    if stability_window is None:
        stability_window = int(sim_cfg.memory_window)
    stability_window = int(stability_window)

    target_rounds = initial_rounds
    states = np.full((target_rounds, sim_cfg.n_agents), np.nan, dtype=np.float32)
    init, initial_majority_opinion = generate_initial_opinions(
        sim_cfg.n_agents,
        sim_cfg.initial_majority_ratio,
        seed_distribution=sim_cfg.seed_distribution,
        distribution_mode=sim_cfg.initial_distribution_mode,
    )
    states[0, :] = init.astype(np.float32)

    neighbor_map = _neighbor_index_map(sim_cfg.n_agents, sim_cfg.n_neighbors)
    max_inflight = max(1, min(len(model_pool), sim_cfg.n_agents))
    session_local = threading.local()
    session_lock = threading.Lock()
    sessions_to_close: List[requests.Session] = []

    requests_log_f = None
    requests_log_buffer: List[str] = []
    requests_log_flush_every = 64
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Overwrite to keep re-runs reproducible (no accidental concatenation).
        requests_log_f = open(output_dir / "requests.jsonl", "w", encoding="utf-8")

    def flush_requests_log_buffer() -> None:
        if requests_log_f is None or not requests_log_buffer:
            return
        requests_log_f.write("".join(requests_log_buffer))
        requests_log_buffer.clear()

    def append_request_log_record(round_idx: int, result: _PromptJobResult) -> None:
        if requests_log_f is None:
            return
        rec = {
            "round": round_idx,
            "agent": result.agent_idx,
            "seed_distribution": sim_cfg.seed_distribution,
            "memory_window": sim_cfg.memory_window,
            "attempts": req_cfg.max_attempts,
            "payload": {k: v for k, v in (result.payload or {}).items() if k != "input"},
            "system_prompt": result.system_prompt,
            "user_prompt": result.user_prompt,
            "raw_response": result.raw_response,
            "parsed_token": result.parsed_token,
            "error": result.error,
            "request_elapsed_s": result.request_elapsed_s,
            "response_model": result.response_model,
            "http_status": result.http_status,
            "usage": result.usage,
            "selected_model": result.selected_model,
            "model_pool": list(model_pool),
        }
        requests_log_buffer.append(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_thread_session() -> requests.Session:
        session = getattr(session_local, "session", None)
        if session is None:
            session = requests.Session()
            session_local.session = session
            with session_lock:
                sessions_to_close.append(session)
        return session

    def run_prompt_job(job: _PromptJob, selected_model: str) -> _PromptJobResult:
        token: Optional[str] = None
        raw_response = ""
        last_err: Optional[str] = None
        last_usage: Optional[Dict[str, Any]] = None
        last_request_elapsed_s: Optional[float] = None
        last_response_model: Optional[str] = None
        last_http_status: Optional[int] = None
        last_payload: Optional[Dict[str, Any]] = None
        actual_user_prompt = append_no_think_if_needed(job.user_prompt, selected_model)

        for attempt in range(1, req_cfg.max_attempts + 1):
            req_t0 = time.time()
            try:
                result = call_llm_responses(
                    base_url=req_cfg.base_url,
                    model=selected_model,
                    system_prompt=job.system_prompt,
                    user_prompt=actual_user_prompt,
                    temperature=req_cfg.temperature,
                    seed=req_cfg.request_seed,
                    max_output_tokens=req_cfg.max_tokens,
                    timeout_s=req_cfg.timeout_s,
                    top_k=req_cfg.top_k,
                    top_p=req_cfg.top_p,
                    min_p=req_cfg.min_p,
                    repeat_penalty=req_cfg.repeat_penalty,
                    stop_regex=_stop_regex,
                    session=get_thread_session(),
                    api_format=req_cfg.api_format,
                )
                last_http_status = result.get("http_status")
                last_request_elapsed_s = result.get("request_elapsed_s")
                last_usage = result.get("usage")
                last_response_model = result.get("response_model")
                last_payload = result.get("payload")
                raw_response = str(result.get("raw_response") or "").strip()
                token = parse_llm_response_token(
                    raw_response,
                    token0,
                    token1,
                    prompt_variant=sim_cfg.prompt_variant,
                )
                if token in (token0, token1):
                    break
                last_err = f"invalid_token attempt={attempt} raw={raw_response[:80]!r}"
            except Exception as e:
                last_request_elapsed_s = time.time() - req_t0
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(0.2)

        return _PromptJobResult(
            agent_idx=job.agent_idx,
            selected_model=selected_model,
            system_prompt=job.system_prompt,
            user_prompt=actual_user_prompt,
            raw_response=raw_response,
            parsed_token=token,
            error=last_err if token not in (token0, token1) else None,
            usage=last_usage,
            http_status=last_http_status,
            request_elapsed_s=last_request_elapsed_s,
            payload=last_payload,
            response_model=last_response_model,
        )

    t0 = time.time()
    consensus_round: Optional[int] = None
    consensus_val: Optional[int] = None
    stop_reason: Optional[str] = None
    truncated: bool = False
    truncation_window: Optional[int] = None
    truncation_round: Optional[int] = None
    rounds_completed: int = 0

    try:
        with ThreadPoolExecutor(max_workers=max_inflight) as executor:
            r = 1
            while r < target_rounds:
                prev = states[r - 1]
                round_jobs: List[_PromptJob] = []
                for i in range(sim_cfg.n_agents):
                    left_indices, right_indices = neighbor_map[i]
                    left = [state_to_token(prev[idx], token0, token1) for idx in left_indices if not np.isnan(prev[idx])]
                    right = [state_to_token(prev[idx], token0, token1) for idx in right_indices if not np.isnan(prev[idx])]
                    current_opinion = state_to_token(prev[i], token0, token1)

                    if sim_cfg.conformity_game:
                        memory_enabled = sim_cfg.memory_window > 0
                        neighborhood_list = left + [current_opinion] + right
                        system_prompt = render_conformity_game_system_prompt(
                            sim_cfg.conformity_game_mode,
                            token0,
                            token1,
                            memory_enabled,
                            sim_cfg.prompt_variant,
                        )
                        user_prompt = render_conformity_game_user_prompt(
                            neighborhood_list,
                            current_opinion,
                            token0,
                            token1,
                            memory_enabled,
                            sim_cfg.prompt_variant,
                        )
                    else:
                        assert strategy is not None
                        system_prompt, user_prompt = strategy.build_prompt(
                            left=left,
                            right=right,
                            current_opinion=current_opinion,
                        )

                    mem_block = _format_memory_block_timeline_precomputed(
                        states,
                        left_indices=left_indices,
                        right_indices=right_indices,
                        agent_idx=i,
                        current_round=r,
                        memory_window=sim_cfg.memory_window,
                        token0=token0,
                        token1=token1,
                    )
                    if mem_block:
                        user_prompt = mem_block + "\n\n=== CURRENT ROUND ===\n\n" + user_prompt

                    round_jobs.append(
                        _PromptJob(
                            agent_idx=i,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    )

                round_results: Dict[int, _PromptJobResult] = {}
                active_futures: Dict[Any, int] = {}
                next_job_idx = 0
                initial_lanes = min(max_inflight, len(round_jobs))

                for lane_idx in range(initial_lanes):
                    job = round_jobs[next_job_idx]
                    next_job_idx += 1
                    active_futures[executor.submit(run_prompt_job, job, model_pool[lane_idx])] = lane_idx

                while active_futures:
                    done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        lane_idx = active_futures.pop(future)
                        result = future.result()
                        round_results[result.agent_idx] = result

                        if result.parsed_token not in (token0, token1):
                            append_request_log_record(r, result)
                            flush_requests_log_buffer()
                            for pending_future in active_futures:
                                pending_future.cancel()
                            raise RuntimeError(
                                "LLM parsing failed after "
                                f"{req_cfg.max_attempts} attempts: "
                                f"round={r} agent={result.agent_idx} "
                                f"model={result.selected_model} error={result.error}"
                            )

                        if next_job_idx < len(round_jobs):
                            job = round_jobs[next_job_idx]
                            next_job_idx += 1
                            active_futures[executor.submit(run_prompt_job, job, model_pool[lane_idx])] = lane_idx

                for job in round_jobs:
                    result = round_results[job.agent_idx]
                    token = result.parsed_token

                    if token == token0:
                        states[r, job.agent_idx] = 0.0
                    elif token == token1:
                        states[r, job.agent_idx] = 1.0
                    else:
                        # keep as NaN (same semantics as Streamlit runner on failure)
                        states[r, job.agent_idx] = np.nan

                    append_request_log_record(r, result)
                    if len(requests_log_buffer) >= requests_log_flush_every:
                        flush_requests_log_buffer()

                ok, val = check_consensus(states[r])
                if ok:
                    consensus_round = r
                    consensus_val = val
                    stop_reason = "consensus"
                    break

                # Stability check: if the last W rounds are identical, the next round would be fully redundant.
                if stability_window >= 2:
                    check_now = True
                    if sim_cfg.stability_check_only_at_initial_horizon:
                        check_now = (r == (initial_rounds - 1))
                    if check_now and r >= (stability_window - 1):
                        stable = True
                        start = r - (stability_window - 1)
                        for rr in range(start, r):
                            if not np.array_equal(states[rr], states[rr + 1], equal_nan=True):
                                stable = False
                                break
                        if stable:
                            next_round = r + 1
                            if next_round >= states.shape[0]:
                                states = np.vstack(
                                    [states, np.full((1, sim_cfg.n_agents), np.nan, dtype=np.float32)]
                                )
                                target_rounds = max(target_rounds, next_round + 1)
                            states[next_round] = states[r]
                            stop_reason = f"stabilized_{stability_window}"
                            truncated = True
                            truncation_window = int(stability_window)
                            truncation_round = int(next_round)
                            break

                # If we reached the current horizon and the system is still changing, extend by doubling.
                if r == (target_rounds - 1) and target_rounds < max_rounds:
                    new_target = min(max_rounds, target_rounds * 2)
                    extra = new_target - target_rounds
                    if extra > 0:
                        states = np.vstack(
                            [states, np.full((extra, sim_cfg.n_agents), np.nan, dtype=np.float32)]
                        )
                        target_rounds = new_target

                r += 1
    finally:
        flush_requests_log_buffer()
        if requests_log_f is not None:
            requests_log_f.close()
        for session in sessions_to_close:
            try:
                session.close()
            except Exception:
                pass

    elapsed_s = time.time() - t0
    # If we broke early, r is the last executed round; truncation may materialize one redundant round ahead.
    if consensus_round is not None:
        rounds_completed = consensus_round
    else:
        if truncated and truncation_round is not None:
            rounds_completed = min(int(truncation_round), target_rounds - 1)
        elif stop_reason is None:
            # Either stabilized without setting stop_reason (shouldn't happen) or ran out of max rounds.
            # r points to the next round to run; last completed is min(r-1, target_rounds-1).
            rounds_completed = min(r - 1, target_rounds - 1)
        else:
            rounds_completed = min(r, target_rounds - 1)

    if stop_reason is None:
        # Reached the end without consensus and without stability stop.
        stop_reason = "max_rounds_reached" if target_rounds >= max_rounds and rounds_completed >= (target_rounds - 1) else "completed_no_consensus"

    states = states[: rounds_completed + 1]
    result = {
        "states": states,
        "consensus_round": consensus_round,
        "consensus_val": consensus_val,
        "elapsed_s": elapsed_s,
        "initial": init.tolist(),
        "initial_majority_opinion": initial_majority_opinion,
        "stop_reason": stop_reason,
        "truncated": truncated,
        "truncation_window": truncation_window,
        "truncation_round": truncation_round,
        "rounds_completed": rounds_completed,
        "initial_rounds": initial_rounds,
        "final_rounds_allocated": int(target_rounds),
        "max_rounds": int(max_rounds),
        "stability_window": int(stability_window),
    }
    if output_dir is not None:
        (output_dir / "result_meta.json").write_text(
            json.dumps(
                {
                    "consensus_round": consensus_round,
                    "consensus_val": consensus_val,
                    "elapsed_s": elapsed_s,
                    "stop_reason": stop_reason,
                    "truncated": truncated,
                    "truncation_window": truncation_window,
                    "truncation_round": truncation_round,
                    "rounds_completed": rounds_completed,
                    "initial_rounds": initial_rounds,
                    "final_rounds_allocated": int(target_rounds),
                    "max_rounds": int(max_rounds),
                    "stability_window": int(stability_window),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return result
