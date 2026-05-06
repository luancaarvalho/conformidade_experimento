#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

DistributionMode = Literal["ratio", "half_split"]


@dataclass(frozen=True)
class InitialDistribution:
    opinions: np.ndarray
    initial_majority_opinion: int
    count_0: int
    count_1: int
    majority_ratio: float
    mode: DistributionMode


def clamp_ratio(majority_ratio: float) -> float:
    return max(0.0, min(1.0, float(majority_ratio)))


def compute_majority_counts(n_agents: int, majority_ratio: float) -> Tuple[int, int]:
    ratio = clamp_ratio(majority_ratio)
    if ratio > 0.5:
        majority_count = int(math.ceil(n_agents * ratio))
        min_majority = n_agents // 2 + 1
        if majority_count < min_majority:
            majority_count = min_majority
    else:
        majority_count = int(math.floor(n_agents * ratio))
    majority_count = max(0, min(n_agents, majority_count))
    minority_count = n_agents - majority_count
    return majority_count, minority_count


def generate_initial_distribution(
    *,
    n_agents: int,
    seed_distribution: int,
    majority_ratio: float = 0.51,
    mode: DistributionMode = "ratio",
) -> InitialDistribution:
    if n_agents <= 0:
        raise ValueError("n_agents must be > 0")

    mode = str(mode).strip().lower()
    if mode not in ("ratio", "half_split"):
        raise ValueError(f"invalid mode={mode!r}; expected 'ratio' or 'half_split'")

    seed_int = int(seed_distribution)
    py_rng = random.Random(seed_int)
    np_rng = np.random.default_rng(seed_int)

    # Deterministic and balanced-majority rule across sequential seeds:
    # odd seed -> majority 1, even seed -> majority 0.
    # Example: seeds 1..30 => 15 with majority 0 and 15 with majority 1.
    majority_val = int(seed_int % 2)

    if mode == "half_split":
        if n_agents % 2 != 0:
            raise ValueError("half_split mode requires even n_agents")
        count_0 = n_agents // 2
        count_1 = n_agents // 2
        opinions = np.concatenate(
            [
                np.full(count_0, 0, dtype=np.int8),
                np.full(count_1, 1, dtype=np.int8),
            ]
        )
        np_rng.shuffle(opinions)
        return InitialDistribution(
            opinions=opinions,
            initial_majority_opinion=majority_val,
            count_0=count_0,
            count_1=count_1,
            majority_ratio=0.5,
            mode="half_split",
        )

    ratio = clamp_ratio(majority_ratio)
    majority_count, minority_count = compute_majority_counts(n_agents, ratio)
    if majority_val == 0:
        opinions = np.concatenate(
            [
                np.full(majority_count, 0, dtype=np.int8),
                np.full(minority_count, 1, dtype=np.int8),
            ]
        )
    else:
        opinions = np.concatenate(
            [
                np.full(majority_count, 1, dtype=np.int8),
                np.full(minority_count, 0, dtype=np.int8),
            ]
        )
    np_rng.shuffle(opinions)
    count_0 = int(np.sum(opinions == 0))
    count_1 = int(np.sum(opinions == 1))
    return InitialDistribution(
        opinions=opinions,
        initial_majority_opinion=majority_val,
        count_0=count_0,
        count_1=count_1,
        majority_ratio=ratio,
        mode="ratio",
    )
