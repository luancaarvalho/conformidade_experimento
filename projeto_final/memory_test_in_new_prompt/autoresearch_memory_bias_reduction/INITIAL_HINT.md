# Initial hint: memory prompt bias reduction

We already solved the structural parsing problem for `v21_zero_shot_cot` with memory and preserved the relaxed v21 parser. Do not change the parser. The next problem is behavioral: Llama 3.1 8B deterministically converges to `k` in every `seed=1-3`, `memory_w=3/5` run, often by round 2 or 3, while Gemma and Qwen produce mixed final states.

Goal: make the prompt less token-biased while keeping all parser guarantees.

Hard constraints:

- Keep the prompt minimal and neutral.
- Keep the MEMORY/CURRENT ROUND structure.
- Keep `{neighbors_section}` exactly as the input injection point.
- Do not add any explicit decision policy: no majority rule, no count-first rule, no tie-breaker, no side/position rule, no memory rule, no current-opinion rule, and no preference for `k` or `z`.
- Do not teach a deterministic algorithm. The model should choose from the provided input and its own short reasoning.
- Do not alter `llm_sim_runner.py` or the v21 parser.
- Preserve extraction-rule interoperability: the same prompt style without memory must still parse and avoid unilateral collapse in the `k/z` extraction gate.

Recommended search direction:

- Reduce repeated token-label framing that may prime one label.
- Avoid phrases that make the first-mentioned symbol feel like a default.
- Ask the model to make the final token match its reasoning, but do not tell it how to reason.
- Emphasize that both symbols are valid labels and that the answer should reflect the concrete input instance.
- Keep the output contract short: reasoning without brackets, final answer on its own line or final token parseable by the existing relaxed v21 parser.

Acceptance:

- `100%` parse rate.
- No run may end with all 30 agents choosing one token.
- Among valid candidates, prefer higher average final-state entropy and lower maximum token skew.
