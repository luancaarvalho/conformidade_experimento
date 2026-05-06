# Initial Hint: V21 With Memory Parse Stability

Objective: starting from the current cross-model keeper prompt, find the smallest prompt-only change that makes `v21_zero_shot_cot` with the `run_batch` memory layer parse reliably on Llama 3.1 8B.

Current failing setup:
- Model: `meta-llama/Llama-3.1-8B-Instruct` through SGLang `/generate`.
- Endpoint: `http://127.0.0.1:30000`.
- Sampling: `temperature=0.0`, `sampling_seed=42`, `max_new_tokens=3000`, `repetition_penalty=1.0`.
- Prompt variant: `v21_zero_shot_cot`.
- Agents: 30.
- Memory format: timeline from `run_batch`.
- Token pair: `k/z`.

Known failures to fix:
- With the current prompt before adding the old verification line, seed 3 with memory windows 3 and 5 failed at round 2, agent 12 with an incomplete final token:

```text
The previous rounds show a clear trend of 'k' opinions dominating, but in the current round, all opinions are 'k' except for the middle one, which is 'z'. 

[k
```

- After adding only the old verification line, seed 3 with memory windows 3 and 5 failed earlier at round 1, agent 7 with two conflicting answer lines:

```text
The current opinion list is mostly 'k's, and I am in the middle, so I should choose 'k' to maintain the majority.

[k
[z
```

What worked in the older memory prompt stabilization:
- The prompt explicitly separated reasoning from the final token.
- It told the model that the final token must match the reasoning.
- It included a final self-check: before sending, verify that the last non-empty line is exactly one bracketed choice token and contains nothing else.
- It discouraged quoting or restating the opinion list.

Constraints for candidates:
- Preserve the existing memory timeline as context. Do not remove memory.
- Preserve the task: choose one of `k` or `z`.
- Do not add a decision algorithm: no majority rule, count-first rule, tie-break rule, side rule, position rule, current-opinion rule, memory rule, or token preference.
- Do not mention left/right/middle as a rule for choosing. The input may still contain the existing position field from `run_batch`.
- Avoid examples that repeat both `[k]` and `[z]` in the prompt more than necessary.
- Keep reasoning plain text and unbracketed.
- The final non-empty line must be parseable as one choice token. For v21 parsing, `[k]`, `[z]`, `[ k ]`, `[ z ]`, bare `k`, and bare `z` are acceptable, but prefer the bracketed form.
- Do not change sampling, SGLang config, model, or code behavior in candidates.

Preferred first hypothesis:
- Simplify the output contract and remove competing instructions like “exactly two non-empty lines” plus an additional “Then write...” line that may cause the model to emit multiple answer lines.
- Use one short reasoning sentence and one final answer line.
- Keep the old successful self-check, but make it part of one coherent output contract, not an extra duplicated instruction.

Acceptance during autoresearch:
- First pass: seed 3, memory windows 3 and 5 must be fully parseable.
- Smoke pass: seeds 1-3, memory windows 3 and 5 must be fully parseable.
- If those pass, the candidate can be promoted for a later full run over seeds 1-20, memory windows 3 and 5.
