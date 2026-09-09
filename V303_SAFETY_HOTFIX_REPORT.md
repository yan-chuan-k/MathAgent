# MathAgent V3.0.3 Safety Hotfix Report

## Release invariants

- Default preset remains `tool_augmented_v31_extended`.
- `tool_augmented_v30` remains the rollback preset.
- Normal execution makes two model calls; a correction is capped at one third
  call and is available only for `ANSWER_VALUE` tasks.
- Formalizer and final-solver sampling settings are unchanged.
- Active tool and verification work remains in the fixed
  `math_agent_core.tools.augmented_worker` subprocess, with no active fork.

## Target-safe correction

Correction eligibility is derived only from the existing grounded
`request_spans` and `requested_actions`.  The parent sends the worker only a
closed enum of targets:

```text
pure_arithmetic
equation_solution
equation_solution_set
matrix_determinant
matrix_rank
```

No target means no verification subprocess and no correction call.  Equation
checks require an explicit solve/roots/solutions/bare-variable request; matrix
checks inspect only the requested matrix noun phrase, not background
conditions such as a matrix being full-rank or having a stated determinant.

## Sound evidence and telemetry

- Polynomial root facts require returned multiplicities to sum to the
  polynomial degree.
- Unevaluated `Integral`, `Sum`, `Limit`, `Derivative`, and `ConditionSet`
  objects are rejected as deterministic facts.
- Empty list results from ambiguous stationary/system solve paths are omitted.
- Complete facts are bounded as a whole; they are never ellipsis-truncated.
  Oversized facts are skipped without preventing later short facts from being
  injected.
- Evidence metadata records which facts actually fit.  Bounded trace fields
  record protocol usability, requested/succeeded/injected operation names,
  worker status, correction targets, and decisive failure methods without raw
  prompts, responses, or equations.
- Stars-and-bars requests require and display `minimum_each=0|1`.

## Local validation

```text
python -m pytest -q
1223 passed in 93.98s

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

These are local safety and integration results only.  They do not establish a
real Judge accuracy result.
