# Memory V21 Autoresearch

This folder contains the isolated autoresearch for validating the cross-model keeper prompt after adding the `run_batch` memory timeline.

The runner starts from `../prompt_templates_memory_test.yaml`, proposes prompt-only changes for `v21_zero_shot_cot`, and evaluates candidates against the known memory failure family before running a broader smoke.

It does not edit `prompt_templates.yaml` production files. Every candidate is evaluated through `PROMPT_TEMPLATE_OVERRIDE_YAML`.

Primary target:
- Llama 3.1 8B on SGLang at `http://127.0.0.1:30000`
- `v21_zero_shot_cot`
- agents 30
- seed distribution 3, memory windows 3 and 5 for the targeted failure pass
- seed distributions 1-3, memory windows 3 and 5 for smoke validation

Success criteria:
- targeted pass: 2/2 parseable
- smoke pass: 6/6 parseable
- no production prompt propagation
