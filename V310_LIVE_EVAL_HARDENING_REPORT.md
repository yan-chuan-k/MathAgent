# V3.1.0 Live Evaluation Hardening Report

## Triggering evidence

The first 35-case Intern-S run was a real decision-grade invocation, but it was
not a complete model-accuracy measurement:

- 35/35 domain routes and 35/35 subtype routes matched the fixture.
- 7 cases returned model completions; 28 cases failed with provider timeout or
  connection errors.
- The old evaluator converted those failures to `无法确定`, graded them as
  wrong answers, and still reported `decision_grade: true`.
- The first completed answer, `103 (9! = 362880)`, was mathematically correct,
  but the grader did not recognize the answer-first value before its
  parenthesized explanation.
- No output truncation was detected. The first case used the targeted
  correction call and retained the corrected answer.

The five completed numeric cases are all correct after the answer extractor
fix. The observed `4/20` score was therefore not a valid model-accuracy score:
it combined one grader false negative with 15 transport failures among the
20 numeric rows.

## V3.1.0 changes

1. Numeric grading recognizes `103 (...)` and explicit answer-first forms while
   preserving later explicit corrections.
2. A solver trace containing a transport/provider error is recorded as
   `transport_error`, not graded as an incorrect mathematical answer.
3. Non-transport solver failures are recorded separately as `solver_error`.
4. A live run with unmeasured rows is now `live_model_partial` and
   `decision_grade: false`; only a complete run can be decision-grade.
5. The report now separates completed, transport-error, solver-error, and
   unmeasured cases. Numeric accuracy is computed only over completed numeric
   cases, with the unmeasured count shown explicitly.
6. The live client timeout and retry count are configurable through
   `INTERN_API_TIMEOUT` and `INTERN_API_RETRY`; retryable transient failures are
   retried while explicit authentication/bad-request failures are not.

## Interpretation rule

Do not optimize mathematical reasoning from a `live_model_partial` report.
First restore provider availability and rerun the unmeasured cases. Only
completed rows with no solver error should affect model accuracy.

Recommended live smoke test:

```bash
INTERN_API_TIMEOUT=180 INTERN_API_RETRY=3 \
python evaluate_web_hard_stress.py --run-agent --limit 3
```

Then run the full set only after the smoke test completes without transport
errors.
