# V3.0.8 Hardening Notes

This patch keeps `tool_augmented_v307_cross_subject` as the default arm and
adds reliability improvements that are activated only when needed.

## Completion robustness

- `InternS1Client` records the provider `finish_reason` without exposing it in
  the visible answer.
- The agent recognizes high-confidence incomplete output from a length stop,
  empty content, unclosed delimiters, ellipsis, or an unfinished connector.
- A complete answer-first line is trusted even when a later explanation is
  cut off.
- Proof, derivation, and construction responses can use one bounded recovery
  call. A failed recovery never discards the original non-empty response.
- Recovery prompts restart from the original problem instead of echoing a large
  partial completion, protecting the remaining context budget.

## Exact subfacts

The V3.0.7 explicit-input tool surface now also supports factorial and
Fibonacci values, primality, divisor counts, Fibonacci modulo a modulus,
geometric PMFs, negative-binomial PMFs, and binomial CDFs. These operations
remain closed-world, bounded, and provenance-bearing; they are never used to
infer unstated assumptions.

## Compatibility

Normal successful tasks retain their previous model settings and call count.
The older preset arms remain available, and new deterministic operations are
filtered out of those rollback arms.
