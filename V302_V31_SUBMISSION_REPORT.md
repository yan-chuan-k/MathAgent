# MathAgent V3.0.2 / V3.1 Submission Report

## Release arm

- Default preset: `tool_augmented_v31_extended`
- Rollback preset: `tool_augmented_v30`
- Production mode: `tool_augmented`
- Extended deterministic tools: enabled
- Normal model calls: 2
- Maximum model calls: 3, only after a decisive exact failure on `ANSWER_VALUE`

## Isolation boundary

Tool batches and system-inferred answer checks are sent as JSON stdin data to
the fixed command:

```text
python -m math_agent_core.tools.augmented_worker
```

The parent uses one absolute monotonic deadline covering startup, transfer,
computation, result transfer, and shutdown. On timeout it terminates, waits
briefly, kills if necessary, reaps the worker, and continues without tool or
verification evidence. The active V3 parent path contains no multiprocessing
fork and calls no legacy ThreadPool verifier.

## Grounding and extended-tool changes

- Exact evaluated inputs remain visible in every emitted deterministic fact.
- Recurrence evidence now states `initial=[a0,...]`, coefficient order, the
  instantiated recurrence, constant, and requested index.
- Eulerian evidence places `connected=true/false` in the request section as a
  formalizer-supplied assumption.
- Result-changing sequences are rejected when oversized instead of truncated:
  CRT lists, recurrence lists, graph degree lists, interpolation points, and
  solve-system equations/variables.
- The extended preset enables number theory, combinatorics, recurrences,
  generating-function coefficients, graph helpers, Newton/interpolation, and
  extended matrix operations.

## Preserved inference settings

| Stage | Temperature | Top-p | Max tokens | Thinking |
| --- | ---: | ---: | ---: | --- |
| Formalizer | 0.2 | 0.95 | 1536 | off |
| Final solver / correction | 0.8 | 0.95 | 32768 | on |

## Validation evidence

```text
python -m pytest -q
1203 passed in 244.25s

python -m compileall -q .
PASS

python diagnose_hard_cases.py --mock --run-agent --production-mode tool_augmented
total=18
route_hits=18
model_calls=36
calls_per_problem=2.000
extended_tools=true
candidate_b=0
critic=0
repair=0
```

The suite ran with normal pytest plugin loading. It includes a 20/20
multi-threaded-parent subprocess stress test, 3.2-second and 20-second worker
attacks against a 2-second deadline, verification-timeout continuation, worker
reaping, fixed-command inspection, closed-world safety attacks, conservative
correction triggers, strict sequence rejection, recurrence semantics, and all
requested extended-tool families.

No real Judge accuracy claim is made by these local regressions. The retained
real baseline is 12/112; the purpose of this package is the next Judge run of
the extended two-stage submission arm.
