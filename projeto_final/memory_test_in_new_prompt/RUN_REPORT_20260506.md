# Memory Layer Test Report - 2026-05-06

## Branch And Setup

- Branch: `memory_test_in_new_prompt`
- Base: `origin/extracao_regras_sglang`
- Worktree on RTX: `/home/ncdia/luan/worktrees/memory_test_in_new_prompt`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Endpoint: `http://127.0.0.1:30000/v1`
- API: SGLang `/generate`
- Prompt override:
  `/home/ncdia/luan/worktrees/memory_test_in_new_prompt/projeto_final/memory_test_in_new_prompt/prompt_templates_memory_test.yaml`
- Variant: `v21_zero_shot_cot`
- Agents: `30`
- Neighbors: `7`
- Request seed: `42`
- Sampling: `temperature=0.0`, `max_new_tokens=3000`, `repetition_penalty=1.0`

## Smoke Results

Smoke 1 output:
`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_in_new_prompt_smoke1_20260506_142517`

Smoke 2 output:
`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_in_new_prompt_smoke2_20260506_142733`

Both smokes passed:

- `4/4` combinations completed in each smoke.
- `0` parse errors in each smoke.
- All `states.npy` hashes matched exactly between smoke 1 and smoke 2.
- All smoke runs converged to `k` at round 2.

Smoke convergence:

```text
seed=1, memory_w=3: FAIL_k_R2
seed=1, memory_w=5: FAIL_k_R2
seed=2, memory_w=3: SUCCESS_k_R2
seed=2, memory_w=5: SUCCESS_k_R2
```

## Full Run Result

Full output:
`/home/ncdia/luan/projeto_final/streamlit_test/batch_outputs/rtx6000/memory_test_in_new_prompt_full_20260506_142953`

The full run was stopped after the first structural parse failure family appeared.

Completed before failure:

```text
seed=1, memory_w=3: ran
seed=1, memory_w=5: ran
seed=2, memory_w=3: ran
seed=2, memory_w=5: ran
```

Failures:

```text
seed=3, memory_w=3: parse failure at round=2, agent=12
seed=3, memory_w=5: same parse failure at round=2, agent=12
```

## Failure Family

The failure is not a wrong semantic choice. It is an incomplete final token:

```text
The previous rounds show a clear trend of 'k' opinions dominating, but in the current round, all opinions are 'k' except for the middle one, which is 'z'. 

[k
```

Expected parseable final forms under the current v21 parser would include `[k]`, `[ k ]`, or bare `k`. The observed output is incomplete because it starts a bracketed token and omits the closing bracket.

The same failure reproduced for `memory_w=3` and `memory_w=5` because at round 2 both memory windows include the same available rounds:

```text
Round 0: ['z', 'z', 'k', 'z', 'z', 'z', 'z']  (you are in the middle)
Round 1: ['k', 'k', 'k', 'z', 'z', 'z', 'z']  (you are in the middle)

Complete Opinion List: ['k', 'k', 'k', 'z', 'z', 'z', 'z']
Your Position: You are the opinion in the middle
Your Current Opinion: z
```

## Interpretation

Adding the timeline memory layer to the keeper prompt passes deterministic smoke testing but does not yet pass the full `run_batch` validation. The first full-run failure is a formatting completion issue triggered by a memory/current-round pattern where reasoning becomes longer and the final answer begins as `[k` without the closing bracket.

Next step should be a focused prompt or stop-regex investigation for this single memory-induced incomplete-bracket family before expanding to all tokens or other models.
