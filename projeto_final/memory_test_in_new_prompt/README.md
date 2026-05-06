# Memory Test In New Prompt

## Purpose

Validate whether the cross-model keeper prompt remains parse-stable when the current `run_batch` timeline memory layer is prepended to the prompt.

## Branch

- Base: `origin/extracao_regras_sglang`
- Branch: `memory_test_in_new_prompt`

## Fixed Decisions

- Initial model: Llama 3.1 8B via SGLang at `http://127.0.0.1:30000`.
- Initial variant: `v21_zero_shot_cot` with k/z tokens.
- Memory format: existing `run_batch` timeline block.
- Primary acceptance metric: parse rate only.
- Smoke: seeds `1-2`, memory windows `3 5`, agents `30`, repeated twice.
- Full run: seeds `1-20`, memory windows `3 5`, agents `30`, only after both smokes parse 100%.
- Parser policy: v9 strict; v21 accepts final lines `[k]`, `[z]`, `k`, `z`, `[ k ]`, `[ z ]`.

## Prompt Override

This branch uses `PROMPT_TEMPLATE_OVERRIDE_YAML` to avoid changing production `prompt_templates.yaml`.

```bash
export PROMPT_TEMPLATE_OVERRIDE_YAML=/home/ncdia/luan/worktrees/memory_test_in_new_prompt/projeto_final/memory_test_in_new_prompt/prompt_templates_memory_test.yaml
```

## Smoke Command Template

```bash
cd /home/ncdia/luan/worktrees/memory_test_in_new_prompt/projeto_final/streamlit_test
python3 run_batch_png.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api-format sglang \
  --prompt-variant v21_zero_shot_cot \
  --agents 30 \
  --neighbors 7 \
  --seeds-distribution 1-2 \
  --memory-windows 3 5 \
  --temperature 0.0 \
  --request-seed 42 \
  --max-tokens 3000 \
  --repeat-penalty 1.0 \
  --mode overwrite \
  --output-dir /home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_in_new_prompt_<timestamp>
```

## Failure Policy

If any smoke has a parse failure, stop before full run and save the failing `requests.jsonl` records with seed, memory window, round, agent, prompt, raw response, and failure family.
