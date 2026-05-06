from __future__ import annotations

from typing import List

CONFORMITY_GAME_MODE_A = "Normative DCT Convergence"
CONFORMITY_GAME_MODE_B = "DCT Consensus Protocol"

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A = (
    "You are playing a Conformity Game inside a small social group.\n\n"
    "The episode has a correct consensus target:\n"
    "the symbol that was more prevalent in the population at the beginning.\n"
    "Agreement on the correct target improves coordination and reduces friction.\n\n"
    "You only see local snapshots and an optional rolling memory. Your objective is to infer the correct target\n"
    "from repeated local evidence and align your public output to accelerate stable convergence.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A_NO_MEMORY = (
    "You are playing a Conformity Game inside a small social group.\n\n"
    "The episode has a correct consensus target:\n"
    "the symbol that was more prevalent in the population at the beginning.\n"
    "Agreement on the correct target improves coordination and reduces friction.\n\n"
    "You only see local snapshots. Your objective is to infer the correct target\n"
    "from repeated local evidence and align your public output to accelerate stable convergence.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B = (
    "You are a node in a local-to-global density-based consensus protocol.\n\n"
    "Each round you must output exactly one symbol: [{T0}] or [{T1}].\n"
    "The correct final consensus is the symbol that was globally more frequent at the beginning of the episode.\n"
    "However you do not see the full initial state. You only receive a local neighborhood snapshot and an optional rolling memory.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B_NO_MEMORY = (
    "You are a node in a local-to-global density-based consensus protocol.\n\n"
    "Each round you must output exactly one symbol: [{T0}] or [{T1}].\n"
    "The correct final consensus is the symbol that was globally more frequent at the beginning of the episode.\n"
    "However you do not see the full initial state. You only receive a local neighborhood snapshot.\n\n"
    "IMPORTANT OUTPUT RULE:\n"
    "Reply with ONLY one token in square brackets: [{T0}] or [{T1}]. No extra text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_COT_MODE_A = (
    "You are playing a Conformity Game inside a small social group.\n\n"
    "The episode has a correct consensus target:\n"
    "the symbol that was more prevalent in the population at the beginning.\n"
    "Agreement on the correct target improves coordination and reduces friction.\n\n"
    "You only see local snapshots and an optional rolling memory. Your objective is to infer the correct target\n"
    "from repeated local evidence and align your public output to accelerate stable convergence.\n\n"
    "After receiving the visible opinions, you must respond with your reasoning and then your choice.\n\n"
    "IMPORTANT: Write your reasoning as plain text only.\n"
    "Do not use markdown, bullet points, labels, or section headers such as 'Reasoning', 'Key Factors', or 'Final Choice'.\n"
    "Write one short plain-text explanation of your decision.\n\n"
    "After completing your reasoning, append a line containing exactly one bracketed choice token: either [{T0}] or [{T1}]. "
    "This must be the very last line of your response and contain ONLY the bracketed token with no additional text."
)

CONFORMITY_GAME_SYSTEM_PROMPT_COT_MODE_B = (
    "You are a node in a local-to-global density-based consensus protocol.\n\n"
    "The correct final consensus is the symbol that was globally more frequent at the beginning of the episode.\n"
    "However you do not see the full initial state. You only receive a local neighborhood snapshot and an optional rolling memory.\n\n"
    "After receiving the visible opinions, you must respond with your reasoning and then your choice.\n\n"
    "IMPORTANT: Write your reasoning as plain text only.\n"
    "Do not use markdown, bullet points, labels, or section headers such as 'Reasoning', 'Key Factors', or 'Final Choice'.\n"
    "Write one short plain-text explanation of your decision.\n\n"
    "After completing your reasoning, append a line containing exactly one bracketed choice token: either [{T0}] or [{T1}]. "
    "This must be the very last line of your response and contain ONLY the bracketed token with no additional text."
)

CONFORMITY_GAME_USER_TEMPLATE_WITH_MEMORY = (
    "Below you can see the current complete list of opinions from your shown neighborhood (including your own).\n\n"
    "**Complete Opinion List:** {NEIGHBORHOOD_LIST}\n"
    "**Your Position:** You are the opinion in the middle\n"
    "**Your Current Opinion:** '{CURRENT_OPINION}'\n\n"
    "Reply ONLY with [{T0}] or [{T1}]."
)

CONFORMITY_GAME_USER_TEMPLATE_NO_MEMORY = (
    "Below you can see the current complete list of opinions from your shown neighborhood (including your own).\n\n"
    "**Complete Opinion List:** {NEIGHBORHOOD_LIST}\n"
    "**Your Position:** You are the opinion in the middle\n"
    "**Your Current Opinion:** '{CURRENT_OPINION}'\n\n"
    "Reply ONLY with [{T0}] or [{T1}]."
)

CONFORMITY_GAME_USER_TEMPLATE_COT = (
    "Below you can see the current complete list of opinions from your shown neighborhood (including your own).\n\n"
    "Complete Opinion List: {NEIGHBORHOOD_LIST}\n"
    "Your Position: You are the opinion in the middle\n"
    "Your Current Opinion: {CURRENT_OPINION}\n\n"
    "Here are the principles to follow when making your choice:\n"
    "1. You must choose an opinion to support: either '{T0}' or '{T1}'\n"
    "2. You should consider the opinions of all visible participants when making your choice\n"
    "3. You can either keep your current opinion or change it based on the group\n"
    "4. Your objective is to help the group converge toward the correct global target suggested by the available evidence\n\n"
    "Please think step by step and write a short plain-text explanation of your decision.\n"
    "Do not use markdown, bullet points, labels, or section headings.\n"
    "Do not repeat your explanation in multiple sections.\n\n"
    "Then end with your bracketed choice on its own line as the final output: [{T0}] or [{T1}]. "
    "Do not add any text after the bracket."
)


def _is_cot_prompt_variant(prompt_variant: str) -> bool:
    return str(prompt_variant or "").strip().lower().startswith("v21_")


def render_conformity_game_system_prompt(
    mode: str,
    token0: str,
    token1: str,
    memory_enabled: bool,
    prompt_variant: str,
) -> str:
    if _is_cot_prompt_variant(prompt_variant):
        tpl = (
            CONFORMITY_GAME_SYSTEM_PROMPT_COT_MODE_B
            if mode == CONFORMITY_GAME_MODE_B
            else CONFORMITY_GAME_SYSTEM_PROMPT_COT_MODE_A
        )
    elif mode == CONFORMITY_GAME_MODE_B:
        tpl = (
            CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B
            if memory_enabled
            else CONFORMITY_GAME_SYSTEM_PROMPT_MODE_B_NO_MEMORY
        )
    else:
        tpl = (
            CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A
            if memory_enabled
            else CONFORMITY_GAME_SYSTEM_PROMPT_MODE_A_NO_MEMORY
        )
    return tpl.format(T0=token0, T1=token1)


def render_conformity_game_user_prompt(
    neighborhood_list: List[str],
    current_opinion: str,
    token0: str,
    token1: str,
    memory_enabled: bool,
    prompt_variant: str,
) -> str:
    if _is_cot_prompt_variant(prompt_variant):
        tpl = CONFORMITY_GAME_USER_TEMPLATE_COT
    else:
        tpl = (
            CONFORMITY_GAME_USER_TEMPLATE_WITH_MEMORY
            if memory_enabled
            else CONFORMITY_GAME_USER_TEMPLATE_NO_MEMORY
        )
    return tpl.format(
        NEIGHBORHOOD_LIST=repr(neighborhood_list),
        CURRENT_OPINION=current_opinion,
        T0=token0,
        T1=token1,
    )
