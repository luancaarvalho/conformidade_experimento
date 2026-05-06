# Qwen Memory V21 Autoresearch State

## Trigger

The full cross-model run with the shortened memory prompt passed Llama and progressed on Gemma, but Qwen produced many parse failures.

Observed full run:

`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_full_crossmodel_qwenfix_20260506_181009`

At interruption:

- Llama 3.1 8B: 40/40 complete, 0 failures.
- Gemma 4B: 34/40 complete, 0 failures.
- Qwen 4B `/no_think`: 14 complete, 25 failures.

The Qwen failures were preserved in:

`/home/ncdia/luan/worktrees/memory_test_in_new_prompt/projeto_final/memory_test_in_new_prompt/qwen_memory_autoresearch_failures_20260506.jsonl`

## Finding

The Qwen failures were usually semantically valid but format-invalid under the previous parser. Typical raw outputs:

```text
<think>

</think>

The middle position is currently z, and the surrounding opinions show a pattern of z's on the sides with k's in the second and third positions, suggesting a possible shift towards k in the next round. z
```

All 25 saved failures became parseable under a minimal v21-only parser relaxation:

- Strip empty `<think></think>`.
- Inspect the final non-empty line.
- Keep existing acceptance for `[k]`, `[z]`, `[ k ]`, `[ z ]`, bare `k`, bare `z`.
- Additionally accept a final standalone plain token at the end of the final line, e.g. `... z` or `... k`.

This is intentionally not applied to v9.

## Prompt State

The current prompt remains the shortened Qwen-compatible memory prompt:

```yaml
v21_zero_shot_cot:
  system: |
    You will participate in an opinion choice activity.

    You must choose exactly one of two symbols: k or z. No other symbols are valid. Use only the provided input fields.
  user: |
    Use the MEMORY section above as prior-round context when it is present. Use the CURRENT ROUND section as the present input. Do not quote or restate the opinion list.

    Input:
    {neighbors_section}

    Write one short sentence of reasoning in plain text only. Do not use square brackets in the reasoning sentence.

    On the next line, write only [k] or [z].
```

## Next Validation

Run full cross-model again with the parser relaxation:

- Llama 3.1 8B, endpoint `127.0.0.1:30000`.
- Gemma 4B, endpoint `127.0.0.1:30001`.
- Qwen 4B, endpoint `127.0.0.1:30002`, `/no_think` auto.
- `seeds_distribution=1-20`.
- `memory_windows=3 5`.
- `agents=30`.

Acceptance remains 40/40 complete and 0 `FAILED.txt` per model.
