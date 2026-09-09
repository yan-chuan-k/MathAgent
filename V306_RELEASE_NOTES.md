# MathAgent V3.0.6 — Cross-Subject Hardened

## Release intent

V3.0.6 keeps the empirically stronger FULL ScoreFirst conditioning and adds
bounded deterministic support for difficult mathematics across the existing
subject router.  It is a controlled improvement over V3.0.5, not a claim of a
new real-Judge score.

The default preset is now:

```python
tool_augmented_v306_cross_subject
```

The frozen rollback control remains available:

```python
tool_augmented_v305_full_final
```

## What changed

- The final solver still uses `temperature=0.8`, `top_p=0.95`,
  `max_tokens=32768`, and `thinking_mode=True`.
- The extractor remains `thinking_mode=False`, `temperature=0.2`,
  `top_p=0.95`, and `max_tokens=1536`.  It is skipped for proof, disproof,
  and construction modes.
- The V3.0.6 final prompt starts with the frozen FULL ScoreFirst prompt.
  Formalizer plans never enter the final prompt.
- A compact high-difficulty discipline is added only for conservative,
  objective complexity signals such as proof/disproof/construction mode,
  multiple requested outputs, long conditioned statements, or advanced
  theorem-language markers.
- Parent-owned automatic tool requests are derived only from request spans,
  never by searching unrelated background premises.  Every injected result
  displays its exact input.
- Added exact, bounded deterministic operations:
  multiplicative order, primitive-root check, partitions into exactly `k`
  parts, binomial/Poisson/hypergeometric PMFs, matrix trace, and bounded
  matrix powers.
- An injected parent-grounded integer/rational fact can trigger a correction
  only when a final ANSWER_VALUE response plainly contradicts it.  Symbolic
  forms, proofs, matrices, and formalizer-selected facts are deliberately not
  subjected to this literal consistency gate.
- V3.0.5 and earlier rollback arms filter V3.0.6-only operations, preserving
  their former deterministic evidence surface.

## Prompt safety invariants

- A standard V3.0.6 task with neither successful deterministic evidence nor a
  high-difficulty card receives byte-identical FULL baseline messages.
- V3.0.5 retains the stronger invariant: with no successful extractor fact,
  its final messages are byte-identical to `full_thinking_on`.
- Tool evidence is optional and provenance-bearing; it never replaces domain
  strategy, subtype guidance, or final verification guidance.
- There is no tool-driven shell, filesystem, network, or arbitrary-code path.

## Validation

Run in the normal pytest plugin environment:

```bash
python -m pytest -q
python -m compileall -q .
python diagnose_hard_cases.py --mock --run-agent --production-mode tool_augmented
```

The release package includes regression coverage for frozen V3.0.5 behavior,
V3.0.6 prompt construction, automatic target grounding, malformed extractor
fallbacks, cross-subject exact tools, and correction gating.

Validated for this package in the normal pytest plugin environment:

- `1270 passed` from `python -m pytest -q`
- `python -m compileall -q .` passed
- Mock hard-case diagnostic: `18/18` route hits, `34` model calls
  (`1.889` per problem), extended tools enabled.  Its mock answer accuracy is
  intentionally not a quality metric because the offline MockClient does not
  solve the problems.

## Real-Judge interpretation

Local tests establish safety and determinism; they do not measure hidden-set
accuracy.  A real-Judge run is still required to determine whether
FULL conditioning plus V3.0.6 grounded computation materially improves on
the known V2.8.1 FULL baseline.
