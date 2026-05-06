# Focused hint: fix residual V21 parse misses only

## Starting point

Start from `crossmodel_iter07_20260505_140125`, the best previous prompt.

This prompt already gets:

- Gemma: v9 and v21 complete, normal and swap.
- Llama: v9 complete; v21 has exactly one parse miss.
- Qwen: v9 complete; v21 has exactly one parse miss.

The full validation still must be run for Gemma, Llama, and Qwen, normal and swap, v9 and v21.

## Exact residual failures

Llama 3.1 8B, v21 swap, config `1010000`, rendered list `kzkzzzz`:

```text
I will choose the symbol that is more common in the list.

k
```

Problem: the final line is a bare symbol, not bracketed.

Qwen 4B /no_think, v21 normal, config `0100110`, rendered list `kzkkzzk`:

```text
The list has seven opinions with 'k' appearing three times and 'z' appearing four times. The middle position is the fourth opinion, which is 'z'.  
[ z ]
```

Problem: the final line has spaces inside the brackets.

## Required search direction

Make the smallest possible V21-only adjustment. Preserve the V9 output contract and shared system/user unless a change is strictly needed.

Good additions:

- the final line has exactly three characters;
- the final line has no spaces before, after, or inside it;
- the final line is invalid if it is a bare symbol without brackets;
- build the final line as opening square bracket, chosen symbol, closing square bracket;
- keep reasoning unbracketed and on a separate line.

Do not add decision rules. Do not add majority, count-first, tie-breaker, side, position, current-opinion, or memory rules.

Do not mention memory, rounds, previous/current context, or history. This is extract-rules, not the memory experiment.

Do not destabilize Gemma or V9: they already work.

