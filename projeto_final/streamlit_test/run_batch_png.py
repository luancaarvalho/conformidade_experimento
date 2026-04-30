#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_sim_runner import LlmRequestConfig, SimulationConfig, check_consensus, run_simulation
from llm_sim_runner import token_pair_for_prompt_variant
from projeto_final.utils.conformity_game_prompts import (
    CONFORMITY_GAME_MODE_A,
    CONFORMITY_GAME_MODE_B,
)


def _parse_int_range(s: str) -> List[int]:
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return [int(s)]
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if not m:
        raise ValueError(f"invalid range: {s!r} (expected N or A-B)")
    a = int(m.group(1))
    b = int(m.group(2))
    if a > b:
        a, b = b, a
    return list(range(a, b + 1))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plot_history_png(
    states: np.ndarray,
    out_path: Path,
    *,
    rule_name: str,
    n_agents: int,
    consensus_round: Optional[int],
    consensus_val: Optional[int],
    success: bool,
    initial_majority: Optional[int],
    token0: str,
    token1: str,
    truncated: bool = False,
    truncation_window: Optional[int] = None,
) -> None:
    n_rounds_completed = states.shape[0]
    # NaNs are treated as neither token0 nor token1.
    percent_0 = np.sum(states == 0.0, axis=1) / n_agents * 100.0
    percent_1 = np.sum(states == 1.0, axis=1) / n_agents * 100.0

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 3]})

    if success:
        title = f"SUCESSO - {rule_name}: {n_agents} Agentes"
        if consensus_round is not None and consensus_val is not None:
            consensus_label = token0 if int(consensus_val) == 0 else token1
            title += f"\nConsenso em {consensus_label} na rodada {consensus_round}"
    else:
        title = f"FALHA - {rule_name}: {n_agents} Agentes"
        if consensus_round is not None and consensus_val is not None:
            consensus_label = token0 if int(consensus_val) == 0 else token1
            majority_label = (
                token0 if int(initial_majority) == 0 else token1
                if initial_majority is not None
                else "?"
            )
            title += f"\nConsenso INCORRETO em {consensus_label} (maioria inicial: {majority_label})"
        else:
            title += f"\nSem Consenso em {n_rounds_completed - 1} Rodadas (limite atual)"
    if truncated:
        suffix = "TRUNCATED"
        if truncation_window is not None:
            suffix += f" (W={int(truncation_window)})"
        title += f"\n{suffix}"

    fig.suptitle(title, fontsize=14, fontweight="bold", color="green" if success else "red")

    axs[0].plot(
        percent_0,
        label=f"% {token0} (0) - Primeiro Token",
        color="lightgray",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=3,
        markevery=max(1, len(percent_0) // 20),
    )
    axs[0].plot(
        percent_1,
        label=f"% {token1} (1) - Segundo Token",
        color="black",
        linestyle="-",
        linewidth=2.5,
        marker="s",
        markersize=3,
        markevery=max(1, len(percent_1) // 20),
    )
    axs[0].set_ylabel("Porcentagem de Agentes")
    axs[0].set_ylim([0, 100])
    axs[0].legend()
    axs[0].grid(True)

    cax = axs[1].imshow(states, cmap="gray_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(cax, ax=axs[1], ticks=[0, 1])
    cbar.set_label("Opinião")
    cbar.set_ticklabels([f"{token0} (0) - Branco/Token1", f"{token1} (1) - Preto/Token2"])

    axs[1].set_xlabel("Agent ID")
    axs[1].set_ylabel("Rodada")
    axs[1].set_xticks(np.arange(0, n_agents, max(1, n_agents // 10)))
    axs[1].set_yticks(np.arange(0, n_rounds_completed, max(1, n_rounds_completed // 10)))

    if consensus_round is not None:
        axs[0].axvline(x=consensus_round, color="black", linestyle=":", linewidth=2)
        axs[1].axhline(y=consensus_round, color="black", linestyle=":", linewidth=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, metadata={})
    plt.close(fig)


def _run_one(
    *,
    base_out: Path,
    req_cfg: LlmRequestConfig,
    sim_base: Dict[str, Any],
    seed_distribution: int,
    memory_window: int,
    mode: str,
) -> Tuple[Path, str, float]:
    t0 = time.time()
    out_dir = base_out / f"seed_distribution_{seed_distribution:04d}" / f"memory_w_{memory_window}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    # "skip" should mean "skip completed". If a previous run was partial, rerun it.
    # This makes the batch idempotent and safe to resume after interruption.
    def _is_complete_run(d: Path) -> bool:
        if not (d / "states.npy").exists():
            return False
        if not (d / "sha256.txt").exists():
            return False
        if not (d / "initial_config.json").exists():
            return False
        if not (d / "result_meta.json").exists():
            return False
        meta_path = d / "result_meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        plot_filename = meta.get("plot_filename")
        if plot_filename:
            if not (d / str(plot_filename)).exists():
                return False
        elif not list(d.glob("sim*.png")):
            return False
        return True

    if mode == "skip" and _is_complete_run(out_dir):
        return out_dir, "skipped", 0.0

    if mode == "overwrite":
        # keep files; overwrite deterministically (we don't delete to avoid accidents)
        pass

    sim_cfg = SimulationConfig(
        **sim_base,
        seed_distribution=seed_distribution,
        memory_window=memory_window,
        # Default behavior requested: stop early if the last W rounds (W=memory_window) are identical.
        stability_window=memory_window,
    )
    token0, token1 = token_pair_for_prompt_variant(sim_cfg.prompt_variant)

    cfg_dump = {
        "request": {
            "base_url": req_cfg.base_url,
            "model": req_cfg.model,
            "model_pool": list(req_cfg.model_pool or ()),
            "temperature": req_cfg.temperature,
            "max_tokens": req_cfg.max_tokens,
            "request_seed": req_cfg.request_seed,
            "timeout_s": req_cfg.timeout_s,
            "max_attempts": req_cfg.max_attempts,
        },
        "simulation": {
            "prompt_variant": sim_cfg.prompt_variant,
            "conformity_game": sim_cfg.conformity_game,
            "conformity_game_mode": sim_cfg.conformity_game_mode,
            "n_agents": sim_cfg.n_agents,
            "n_rounds": sim_cfg.n_rounds,
            "max_rounds": sim_cfg.max_rounds,
            "n_neighbors": sim_cfg.n_neighbors,
            "initial_majority_ratio": sim_cfg.initial_majority_ratio,
            "initial_distribution_mode": sim_cfg.initial_distribution_mode,
            "memory_window": sim_cfg.memory_window,
            "memory_format": sim_cfg.memory_format,
            "seed_distribution": sim_cfg.seed_distribution,
            "stability_window": sim_cfg.stability_window,
        },
    }
    config_path.write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    res = run_simulation(sim_cfg, req_cfg, output_dir=out_dir)
    states = res["states"]

    npy_path = out_dir / "states.npy"
    np.save(npy_path, states)

    init_path = out_dir / "initial_config.json"
    initial = res["initial"]
    initial_majority_opinion = int(res.get("initial_majority_opinion")) if res.get("initial_majority_opinion") is not None else None
    count_0 = int(np.sum(np.array(initial, dtype=np.int8) == 0))
    count_1 = int(np.sum(np.array(initial, dtype=np.int8) == 1))
    init_path.write_text(
        json.dumps(
            {
                "sim_number": seed_distribution,
                "seed": seed_distribution,
                "n_agents": sim_cfg.n_agents,
                "majority_ratio": sim_cfg.initial_majority_ratio,
                "initial_majority_opinion": initial_majority_opinion,
                "initial_config": initial,
                "rodada_0": initial,
                "rodada_0_stats": {
                    "count_0": count_0,
                    "count_1": count_1,
                    "percent_0": (count_0 / sim_cfg.n_agents) * 100.0,
                    "percent_1": (count_1 / sim_cfg.n_agents) * 100.0,
                },
                "opinion_labels": {"0": token0, "1": token1},
                "timestamp": __import__("time").strftime("%Y%m%d_%H%M%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    consensus_round = res["consensus_round"]
    consensus_val = res["consensus_val"]
    truncated = bool(res.get("truncated"))
    truncation_window = res.get("truncation_window")
    truncation_round = res.get("truncation_round")
    ok, val = check_consensus(states[consensus_round] if consensus_round is not None else states[-1])
    _ = ok, val

    rounds_completed = states.shape[0] - 1

    consensus_reached = consensus_round is not None and consensus_val is not None
    correct_consensus = (
        consensus_reached
        and initial_majority_opinion is not None
        and int(consensus_val) == int(initial_majority_opinion)
    )

    if consensus_reached:
        final_consensus_label = token0 if int(consensus_val) == 0 else token1
        status = f"{'SUCCESS' if correct_consensus else 'FAIL'}_{final_consensus_label}_R{consensus_round}"
    else:
        status = f"FAIL_no_consensus_{rounds_completed}R"

    # Nome principal alinhado ao run_automaton_numba.
    png_suffix = "_truncated" if truncated else ""
    png_name = f"sim{seed_distribution:03d}_{status}{png_suffix}.png"
    png_path = out_dir / png_name

    rule_name = f"{sim_cfg.prompt_variant}_sim{seed_distribution:03d}"
    _plot_history_png(
        states,
        png_path,
        rule_name=rule_name,
        n_agents=sim_cfg.n_agents,
        consensus_round=consensus_round,
        consensus_val=consensus_val,
        success=bool(correct_consensus),
        initial_majority=initial_majority_opinion,
        token0=token0,
        token1=token1,
        truncated=truncated,
        truncation_window=truncation_window,
    )

    # Enriquecer result_meta com métricas estilo run_automaton_numba.
    try:
        result_meta_path = out_dir / "result_meta.json"
        if result_meta_path.exists():
            meta = json.loads(result_meta_path.read_text(encoding="utf-8"))
        else:
            meta = {}
        meta.update(
            {
                "consensus_reached": bool(consensus_reached),
                "correct_consensus": bool(correct_consensus),
                "initial_majority_opinion": initial_majority_opinion,
                "final_consensus_opinion": int(consensus_val) if consensus_val is not None else None,
                "status_label": status,
                "status_label_extended": f"{status}{png_suffix}",
                "truncated": truncated,
                "truncation_window": truncation_window,
                "truncation_round": truncation_round,
                "plot_filename": png_name,
            }
        )
        result_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    hashes_path = out_dir / "sha256.txt"
    hashes_path.write_text(
        "\n".join(
            [
                f"states.npy  {_sha256_file(npy_path)}",
                f"{png_name}  {_sha256_file(png_path)}",
            ]
        )
        .strip()
        + "\n",
        encoding="utf-8",
    )
    elapsed_s = float(res.get("elapsed_s") or (time.time() - t0))

    # Root-level timing log for long batches (best-effort).
    try:
        timing_path = base_out / "timings.jsonl"
        with timing_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "seed_distribution": seed_distribution,
                        "memory_window": memory_window,
                        "elapsed_s": elapsed_s,
                        "out_dir": str(out_dir),
                        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass

    return out_dir, "ran", elapsed_s


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch runner (Streamlit-derived) that outputs PNGs per run.")
    p.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://172.18.254.18:1234/v1"))
    p.add_argument("--model", default=os.getenv("LLAMA_MODEL", "google/gemma-3-12b"))
    p.add_argument(
        "--model-pool",
        default=os.getenv("LLM_MODEL_POOL", ""),
        help="Compatibilidade legada. Neste runner, o primeiro alias/modelo é usado de forma fixa por worker.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--request-seed", type=int, default=42)

    p.add_argument("--prompt-variant", default="v9_lista_completa_meio_kz")
    p.add_argument("--agents", type=int, default=10)
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of rounds. If omitted, defaults to --agents (checkpoint at round==agents).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional max rounds. If omitted, defaults to 2*--agents (doubling rule).",
    )
    p.add_argument("--neighbors", type=int, default=7)
    p.add_argument("--initial-majority-ratio", type=float, default=0.51)
    p.add_argument(
        "--initial-distribution-mode",
        choices=["auto", "ratio", "half_split"],
        default=os.getenv("INITIAL_DISTRIBUTION_MODE", "auto"),
    )

    p.add_argument("--seeds-distribution", default="1-20", help="Seed range for initial distribution, e.g. 1-20")
    p.add_argument("--memory-windows", nargs="+", type=int, default=[3, 5])
    p.add_argument("--conformity-game", action="store_true", help="Usa o mesmo prompt de jogo da conformidade da interface v4.")
    p.add_argument(
        "--conformity-game-mode",
        default=CONFORMITY_GAME_MODE_A,
        choices=[CONFORMITY_GAME_MODE_A, CONFORMITY_GAME_MODE_B],
        help="Modo do prompt compartilhado de jogo da conformidade.",
    )
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "batch_outputs" / "macstudio_v9_kz"))
    p.add_argument("--mode", choices=["skip", "overwrite"], default="skip")
    p.add_argument("--api-format", choices=["responses", "sglang"], default="responses",
                   help="API format: 'responses' for LM Studio /v1/responses, 'sglang' for SGLang /generate")
    p.add_argument("--repeat-penalty", type=float, default=None,
                   help="Repetition penalty for sampling (e.g. 1.1 to prevent loops). If omitted, not sent.")
    p.add_argument("--top-k", type=int, default=None,
                   help="top_k for sampling. Use 1 for greedy-equivalent with repetition_penalty applied.")

    args = p.parse_args(argv)

    n_rounds = int(args.rounds) if args.rounds is not None else int(args.agents)
    n_agents = int(args.agents)
    max_rounds_default = 2 * n_agents
    max_rounds = int(args.max_rounds) if args.max_rounds is not None else max_rounds_default
    if max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
    if max_rounds > max_rounds_default:
        raise ValueError(f"--max-rounds cannot exceed 2*--agents (= {max_rounds_default})")
    if n_rounds > max_rounds:
        raise ValueError("--rounds cannot be greater than --max-rounds")

    seeds_distribution: List[int] = []
    for part in str(args.seeds_distribution).split(","):
        seeds_distribution.extend(_parse_int_range(part))

    base_out = Path(args.output_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    model_pool = [m.strip() for m in str(args.model_pool).split(",") if m.strip()]
    if not model_pool:
        model_pool = [str(args.model).strip()]

    _temperature = float(args.temperature)
    _max_attempts = 2 if _temperature == 0.0 else 5
    req_cfg = LlmRequestConfig(
        base_url=args.base_url,
        model=model_pool[0],
        temperature=_temperature,
        max_tokens=int(args.max_tokens),
        model_pool=tuple(model_pool),
        request_seed=int(args.request_seed) if args.request_seed is not None else None,
        api_format=args.api_format,
        repeat_penalty=float(args.repeat_penalty) if args.repeat_penalty is not None else None,
        top_k=int(args.top_k) if args.top_k is not None else None,
        max_attempts=_max_attempts,
    )
    print(f"[CONFIG] model_pool={list(req_cfg.model_pool or ())}")

    sim_base: Dict[str, Any] = {
        "prompt_variant": args.prompt_variant,
        "conformity_game": bool(args.conformity_game),
        "conformity_game_mode": str(args.conformity_game_mode),
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "max_rounds": max_rounds,
        "n_neighbors": int(args.neighbors),
        "initial_majority_ratio": float(args.initial_majority_ratio),
        "initial_distribution_mode": str(args.initial_distribution_mode),
        "memory_format": "timeline",
    }

    run_index = 0
    total = len(seeds_distribution) * len(args.memory_windows)
    for seed_distribution in seeds_distribution:
        for mem_w in args.memory_windows:
            run_index += 1
            try:
                out_dir, status, elapsed_s = _run_one(
                    base_out=base_out,
                    req_cfg=req_cfg,
                    sim_base=sim_base,
                    seed_distribution=seed_distribution,
                    memory_window=int(mem_w),
                    mode=args.mode,
                )
            except RuntimeError as exc:
                elapsed_s = time.time() - (time.time())
                fail_dir = base_out / f"seed_distribution_{seed_distribution:04d}" / f"memory_w_{mem_w}"
                fail_dir.mkdir(parents=True, exist_ok=True)
                (fail_dir / "FAILED.txt").write_text(str(exc), encoding="utf-8")
                print(f"[{run_index}/{total}] seed_distribution={seed_distribution} mem_w={mem_w}: FAILED — {exc}")
                sys.stdout.flush()
                continue
            if status == "skipped":
                print(f"[{run_index}/{total}] seed_distribution={seed_distribution} mem_w={mem_w}: {status} -> {out_dir}")
            else:
                print(
                    f"[{run_index}/{total}] seed_distribution={seed_distribution} mem_w={mem_w}: {status}"
                    f" elapsed={elapsed_s:.1f}s -> {out_dir}"
                )
            sys.stdout.flush()

    print(f"Done. Output: {base_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
