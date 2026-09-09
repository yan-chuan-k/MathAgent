# V3.0.4 post-review patch

This package keeps the V3.0.4 object-grounded correction design and adds a
small reliability layer.

## Changes

- Parent-bound arithmetic, equation, and matrix targets are converted into
  deterministic pre-solve requests even when the formalizer emits no `TOOL`.
- Target requests are stable-deduplicated and placed before advisory
  formalizer requests under the worker request cap.
- Tool telemetry separates formalizer facts from parent-target facts while
  retaining the existing trace keys for compatibility.
- A correction response is rechecked in the same isolated worker. If its
  decisive failures increase, the safer Call-2 answer is retained.
- Verification exceptions fail closed without aborting the solve path; the
  previous legacy-variable reference was removed.
- Unsupported domain-sensitive solution-set checks no longer discard the safe
  point-solution check.

## Local validation

```text
python -m compileall -q .       PASS
python -m pytest -q             1242 passed
python diagnose_hard_cases.py --mock --run-agent --production-mode tool_augmented
18/18 routed; 36 model calls; 2.000 calls/problem
```

The attached historical Judge log is a V2.9 run and is not evidence for this
V3.0.4 package. A new real-Judge run is still required to measure accuracy.
