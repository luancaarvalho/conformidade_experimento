# Baseline repeat: memory prompt parserfix

Captured from:

`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_repeat_1_3_parserfix_20260507_140027`

Compared against:

`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_full_crossmodel_parserfix_20260506_204111`

Configuration:

- Commit: `1168e4c`
- Prompt override: `/home/ncdia/luan/worktrees/memory_test_in_new_prompt/projeto_final/memory_test_in_new_prompt/prompt_templates_memory_test.yaml`
- Variant: `v21_zero_shot_cot`
- Agents: `30`
- Seeds: `1-3`
- Memory windows: `3 5`
- Sampling: `temperature=0.0`, `request_seed=42`, `max_tokens=3000`, `repeat_penalty=1.0`
- SGLang servers used the existing deterministic setup with `--enable-deterministic-inference`, `--disable-radix-cache`, and `--max-running-requests 3`.

## Result summary

All completed cases matched the previous full run by `states.npy` hash.

| Model | Completed | Failures | Hash match |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B | 6/6 | 0 | 6/6 |
| Gemma 4B | 6/6 | 0 | 6/6 |
| Qwen 4B `/no_think` | 6/6 | 0 | 6/6 |

## Convergence and final counts

### Llama 3.1 8B

| seed | memory_w | stop | consensus_round | consensus | final counts |
| ---: | ---: | --- | ---: | --- | --- |
| 1 | 3 | consensus | 3 | k | k=30, z=0 |
| 1 | 5 | consensus | 3 | k | k=30, z=0 |
| 2 | 3 | consensus | 2 | k | k=30, z=0 |
| 2 | 5 | consensus | 2 | k | k=30, z=0 |
| 3 | 3 | consensus | 3 | k | k=30, z=0 |
| 3 | 5 | consensus | 3 | k | k=30, z=0 |

Llama is fully parseable and deterministic, but behaviorally collapsed to `k` in every tested run. This is the primary bias target for the new autoresearch.

### Gemma 4B

| seed | memory_w | stop | consensus | final counts |
| ---: | ---: | --- | --- | --- |
| 1 | 3 | stabilized_3 | no | k=12, z=18 |
| 1 | 5 | stabilized_5 | no | k=14, z=16 |
| 2 | 3 | stabilized_3 | no | k=14, z=16 |
| 2 | 5 | stabilized_5 | no | k=15, z=15 |
| 3 | 3 | stabilized_3 | no | k=20, z=10 |
| 3 | 5 | stabilized_5 | no | k=16, z=14 |

### Qwen 4B `/no_think`

| seed | memory_w | stop | consensus | final counts |
| ---: | ---: | --- | --- | --- |
| 1 | 3 | stabilized_3 | no | k=22, z=8 |
| 1 | 5 | stabilized_5 | no | k=25, z=5 |
| 2 | 3 | stabilized_3 | no | k=25, z=5 |
| 2 | 5 | stabilized_5 | no | k=25, z=5 |
| 3 | 3 | stabilized_3 | no | k=24, z=6 |
| 3 | 5 | stabilized_5 | no | k=26, z=4 |

Qwen is parseable and deterministic, but still has a strong `k` skew. It is a secondary behavioral target after Llama stops being unilateral.
