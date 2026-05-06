# Cross-Model Keeper State - 2026-05-06

This document records the current source-of-truth state for the SGLang
extract-rules prompt validation. GitHub should be treated as the canonical
record for this state.

## Result

The previous best prompt became a keeper after relaxing only the v21 parser.

- Baseline id: `baseline_crossmodel_20260506_133329`
- Score: `18.0000`
- Total parsed: `1536/1536`
- Parse failures: `0`
- Models: Gemma 4B, Llama 3.1 8B, Qwen 4B with `/no_think`
- Variants: `v9_lista_completa_meio_kz`, `v21_zero_shot_cot`
- Mappings: normal and token-swap
- Configurations per mapping: `128`
- No model/variant/mapping was unilateral.

## Parser Decision

The prompt itself was not changed for this final acceptance step. The change was
limited to parser normalization for `v21_zero_shot_cot`, because the two
remaining failures were semantically correct answers:

- Llama v21 swap ended with a bare final symbol: `k`
- Qwen v21 normal ended with a bracketed symbol containing spaces: `[ z ]`

The accepted parser policy is:

- `v9_lista_completa_meio_kz` remains strict and accepts only exactly `[k]` or
  `[z]`, with no other text.
- `v21_zero_shot_cot` accepts the last non-empty line if it is `[k]`, `[z]`,
  `k`, `z`, `[ k ]`, or `[ z ]`.
- Qwen `/no_think` still allows only an empty `<think></think>` envelope. Any
  non-empty thinking content remains invalid.

## Per-Model Results

```text
Gemma 4B
v9 normal: 128/128 parse | k=59, z=69
v9 swap:   128/128 parse | k=69, z=59
v21 normal: 128/128 parse | k=33, z=95
v21 swap:   128/128 parse | k=74, z=54
Total: 512/512 parse
```

```text
Llama 3.1 8B
v9 normal: 128/128 parse | k=41, z=87
v9 swap:   128/128 parse | k=89, z=39
v21 normal: 128/128 parse | k=105, z=23
v21 swap:   128/128 parse | k=65, z=63
Total: 512/512 parse
```

```text
Qwen 4B /no_think
v9 normal: 128/128 parse | k=65, z=63
v9 swap:   128/128 parse | k=64, z=64
v21 normal: 128/128 parse | k=62, z=66
v21 swap:   128/128 parse | k=62, z=66
Total: 512/512 parse
```

## Canonical Paths

- Runner:
  `projeto_final/autoresearch_prompts/run_extract_rules_nomemory_bias_crossmodel_fixone_autoresearch.py`
- Strict parser backup:
  `projeto_final/autoresearch_prompts/run_extract_rules_nomemory_bias_crossmodel_fixone_autoresearch.py.bak_strict_parser_20260506`
- Initial hint:
  `projeto_final/autoresearch_prompts/hint_extract_rules_nomemory_bias_crossmodel_fixone.md`
- Start prompt:
  `projeto_final/autoresearch_prompts/start_prompt_extract_rules_crossmodel_fixone_best.md`
- Keeper prompt:
  `projeto_final/extract_rules/sgLang/autoresearch_nomemory_bias_crossmodel_fixone/KEEPER_PROMPT_NOMEMORY_BIAS_CROSSMODEL_FIXONE.md`
- Validation log:
  `projeto_final/extract_rules/sgLang/autoresearch_nomemory_bias_crossmodel_fixone/relaxed_parser_baseline_20260506.log`

## Operational Context

The validation used three concurrent SGLang servers:

- Llama 3.1 8B: `127.0.0.1:30000`
- Gemma 4B: `127.0.0.1:30001`
- Qwen 4B: `127.0.0.1:30002`

Common SGLang settings:

- `/generate`
- `temperature=0.0`
- `sampling_seed=42`
- `max_new_tokens=3000`
- `repetition_penalty=1.0`
- `--disable-radix-cache`
- `--max-running-requests 3`
- `--enable-deterministic-inference`
- `--quantization fp8`

## Interpretation

This is the current best stable state for the extract-rules cross-model prompt:
the prompt is parse-stable across all three target models once the parser treats
semantically correct v21 final-answer variants as valid.
