# MathAgent V3.0.4 Object-Grounded Correction Gate

V3.0.4 preserves the V3.0.3 fork-free fixed subprocess worker and the
`tool_augmented_v31_extended` release preset while replacing type-only answer
verification with parent-bound `VerificationRequest` objects.

## Safety changes

- The parent derives exact equation, matrix, and pure-arithmetic objects from
  grounded request spans. Background equations and matrices are never selected.
- The worker receives only bounded `requests` and revalidates the closed schema
  before evaluation. It does not receive or scan the original problem.
- Equation requests require one parseable equation and an explicit or uniquely
  inferable variable. Matrix requests use a literal target, an exact named
  binding, or one unambiguous pronoun candidate. Ambiguity fails closed.
- Bare-variable matching rejects `x^4`, `x_`, `f(x)`, and other expression
  continuations.
- Correction remains limited to decisive failures from the isolated worker;
  otherwise the Call-2 answer is retained.

## Verification

The V3.0.4 regression suite covers background-vs-target equations, named and
literal matrices, expression-target false positives, exact correction evidence,
closed request schema validation, and subprocess-only verification.

The final package must be checked with normal-plugin `pytest`, `compileall`,
and the 18-case hard diagnostic before Judge use. Local checks do not claim
real-Judge accuracy.
