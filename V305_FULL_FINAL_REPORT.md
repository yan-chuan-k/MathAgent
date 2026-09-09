# MathAgent V3.0.5 — FULL Final + Grounded Deterministic Tools

## Release arm

The production default is now:

```text
tool_augmented_v305_full_final
```

Its settings are:

```text
prompt_profile=full
thinking_mode=true
production_mode=tool_augmented
extended_tools=true
```

`tool_augmented_v31_extended` and `full_thinking_on` remain available as
explicit rollback/comparison arms.

## Controlled change

V3.0.5 restores the frozen FULL ScoreFirst final-solver context and treats the
low-thinking first call strictly as a deterministic tool extractor.

- `ANSWER_VALUE` and `DERIVATION`: extractor call followed by FULL final solver.
- `PROOF`, `PROOF_OR_DISPROOF`, and `CONSTRUCTION_COUNTEREXAMPLE`: FULL final
  solver directly, with one normal model call.
- Extractor plans and advisory prefixes are never injected into the V3.0.5 final
  solver.
- The final prompt is byte-identical to `full_thinking_on` whenever no usable
  deterministic fact exists, including extractor failure, malformed request,
  timeout, plan-only output, or zero requests.
- Successful facts are inserted immediately before the original `Problem:`
  section and retain exact shown-input provenance.
- Extended deterministic tool families remain enabled.

## Reliability changes

- Default answer-verification timeout: `5.0` seconds.
- Direct verification default cap constant: `MAX_VERIFY_TIMEOUT_SECONDS = 5.0`.
- Explicit shorter timeouts remain hard deadlines.
- Unexpected tool-worker exceptions degrade to no-evidence FULL fallback.
- Verification exceptions retain the Call-2 answer; they do not cause a global
  fallback.
- Equation verification requests are extended exactly once.

## Regression evidence

All tests were run with the normal pytest plugin environment.

```text
python -m pytest -q
1257 passed in 135.84s

python -m compileall -q .
PASS

python diagnose_hard_cases.py --mock --run-agent --production-mode tool_augmented
18 / 18 routed, extended_tools=true, 34 model calls
```

The diagnostic now uses 34 rather than 36 calls because its two proof tasks
intentionally skip the extractor and go straight to the FULL final solver.

Dedicated V3.0.5 regressions cover:

- 110 / 110 frozen cases with byte-identical no-tool FULL final messages;
- extractor failure, malformed tool line, plan-only output, and worker timeout
  all falling back to the exact FULL prompt;
- successful evidence as the only prompt augmentation;
- proof/disproof/construction one-call routing;
- incomplete equation and wrong named-matrix correction calls;
- verifier exception retention of the Call-2 response.

## Judge scope

This validates the controlled implementation and local regressions. The real
hidden Judge remains the required measurement for whether FULL conditioning
plus grounded deterministic evidence exceeds the known 12 / 112 baseline.
