# MathAgent V3.0.7 — Cross-Subject Hardening

## Goal

Improve difficult, high-share mathematics items while preserving the V3.0.5
FULL-prompt fallback and reducing visible-answer truncation.

## Default release arm

`tool_augmented_v307_cross_subject`

- FULL ScoreFirst conditioning remains the final-solver base.
- Final solver keeps `thinking_mode=True`, temperature `0.8`, `top_p=0.95`,
  and uses `max_tokens=49152`.
- Proof, disproof, and construction modes skip the low-thinking extractor and
  go directly to the FULL final solver.
- If the extractor or tools produce no usable facts, the final messages remain
  byte-identical to the FULL baseline.
- Deterministic facts are the only final-prompt augmentation; formalizer plans
  are retained only in bounded trace fields.

## Cross-subject additions

The bounded recognizer covers explicit subcomputations commonly occurring in
the larger subject groups:

- Algebra/calculus: coefficient extraction, differentiation, integration,
  limits, and exact matrix linear algebra.
- Probability/statistics: binomial, Poisson, and hypergeometric finite facts.
- Discrete mathematics: partitions, restricted compositions, derangements,
  onto functions, multiset permutations, Catalan structures, recurrences,
  tilings, and lattice paths.
- Graph theory: degree counts, tree identities, planar faces, matchings,
  chromatic numbers, Euler criteria, Hall guarantees, and Turán bounds.
- Number theory: gcd, totient, modular powers, inverses, CRT, linear
  congruences, multiplicative order, primitive roots, and inclusion-exclusion
  counts of multiples.

Every request is bounded, target-oriented, and fail-closed when the explicit
input cannot be uniquely recovered. Tool facts preserve their exact input
provenance and incomplete symbolic results are not injected as evidence.

## Output-length control

The final solver receives a short instruction to use its internal reasoning
budget while emitting only one compact, complete, judgeable solution. It is
not asked to print scratch work, abandoned approaches, repeated checks, or
meta-commentary.

## Validation

Performed in the available runtime:

- `python -m py_compile user_agent.py math_agent_core/tools/augmented_tool.py`
- `python -m compileall -q .`
- V3.0.7 focused tests: 6/6 passed by direct invocation.
- Explicit discrete sample request coverage: 100/100 rows.
- Standard no-fact prompt fallback: byte-identical to `full_thinking_on`.
- Fixed subprocess protocol and bounded combinatorics/graph execution smoke
  checks passed.

The current container did not provide pytest, SymPy, or jsonschema, and package
installation was blocked by network approval. Therefore a complete normal
`pytest` run and real Judge score are not claimed here; run them in the
competition environment before treating this package as accuracy-validated.
