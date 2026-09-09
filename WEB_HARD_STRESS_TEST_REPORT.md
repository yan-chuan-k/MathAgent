# Public Hard-Problem Stress Test Report

## Scope and evidence

The stress set uses 35 concise paraphrases of official public problems. It
contains 29 exact-answer items and 6 proof items. The source papers and answer
keys are linked here:

- [HMMT February 2025 Algebra and Number Theory problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnt/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnt/solutions.pdf)
- [HMMT February 2025 Combinatorics problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/comb/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/comb/solutions.pdf)
- [HMMT February 2025 Geometry problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geo/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geo/solutions.pdf)
- [IMO 2025 official problems](https://www.imo-official.org/assets/documents/problems/2025/2025_eng.pdf) and [official shortlist solutions](https://www.imo-official.org/assets/documents/problems/2025/IMO2025SL.pdf)
- [MAA Putnam archive](https://maa.org/maa-putnam-archive/) was also checked as a separate source catalog for another high-difficulty competition family.

The evaluator sends only the problem, source label, subject hint, task type,
language, difficulty hint, and concept tags. Expected answers, gold summaries,
and grading metadata remain evaluator-only.

## Independent mathematical check

The exact-answer keys were independently checked against the official HMMT
solutions and by short exact derivations:

| Source block | Answers in problem order | Independent check signal |
|---|---|---|
| HMMT A/NT #1–#9 | `103`; `3375`; `1/576`; `-984`; `890`; `1311/2017`; `9/sqrt(23)`; `1-2/pi`; `1037` | divisor factorization; radical divisibility; log-variable products; paired floor terms; minimal polynomial; Euler/Wilson; cotangent triangle; binary digits of `1/pi`; finite-field Lagrange coefficient |
| HMMT Combinatorics #1–#10 | `56`; `29`; `105`; `2304`; `200`; `6300`; `2^25*26!`; `2025/101`; `4/9`; `448/3` | local adjacency forcing; shortest-path complement; pair compression; layer-count product; Eulerian upper/lower bound; interval inclusion-exclusion; triple-block decomposition; boundary indicators; convex-hexagon geometry; conditioned coalescing walk |
| HMMT Geometry #1–#10 | `26`; `63`; `8*sqrt(10)`; `20`; `sqrt(23)-2*sqrt(3)`; `9*sqrt(15)`; `7/18`; `sqrt(6)`; `14+4*sqrt(37)`; `sqrt(95/24)` | law of cosines; midpoint height; four altitude values; nested semicircle similarity; antipode; circumcenter similarity; angle-bisector ratios; power of a point; perpendicular diagonals; centrally symmetric prism section |

The answer forms above are accepted as exact symbolic expressions by the
benchmark grader after changing the relevant grading type from numeric to
symbolic. A self-grade of each key against itself returned `CORRECT`.

## Proof attempts and limits

The six IMO items are intentionally retained as proof-quality stress cases,
not reduced to answer-only checks. The official answer targets are:

- P1: `k in {0,1,3}`.
- P2: the stated parallel line is tangent; a complete coordinate/inversion
  proof is still required.
- P3: `c=4`.
- P4: `a1 = 12^M * 6^N`, with the coprimality condition on `N` from the
  official statement.
- P5: Bazza wins below `1/sqrt(2)`, Alice wins above it, and equality is a
  draw.
- P6: minimum `2112` tiles.

These targets were checked against the official shortlist, but this run does
not claim that the current agent independently completed all six rigorous
proofs. The biggest risks are the nontrivial extremal induction in P1, the
circle configuration in P2, the divisor dynamics in P4, the strategy proof at
the threshold in P5, and the barrier-reef lower bound in P6.

One transcription trap was found and corrected before evaluation: IMO P3 uses
`f(b)^(f(a))`, not the visually collapsed `f(b)*f(a)` that a PDF text
extractor can produce. With the wrong parse the problem appears to force
`f(a)=1`, which is false for the actual problem and produces the wrong bound.

## What was actually runnable here

There is no `INTERN_API_KEY` in this environment, so a live Intern-S model run
could not be honestly performed. The evaluator reports this as
`live_model_blocked`; it does not count `MockClient` output as solving. The
MockClient run is only a pipeline sanity check and intentionally returns
`mock_result`.

The route-only run after the V3.0.9 changes achieved 35/35 domain matches and
35/35 annotated subtype matches. The 5-case mock run completed its pipeline
with 0/5 correct answers, as expected for a non-solving test double.

## Optimizations made from the stress review

1. Short hard statements now receive the compact target/hypothesis ledger and
   independent-check discipline when their caller supplies a coarse contest
   difficulty hint or their wording has multiple high-signal structures.
2. Router overrides recognize divisor-digit/floor-sum number theory, constrained
   circle arrangements, box/letter assignments, and disjoint-rectangle counts;
   explicit “use a recurrence/generating function” instructions still take
   precedence.
3. A factorial appearing only as the exponent in a large modular remainder
   problem is no longer materialized as a useless oversized tool fact.
4. The evaluation package now lazily imports its grader, removing an import
   cycle that made direct strategy-pool imports fail.
5. Provider length metadata, delimiter/connector checks, and one bounded clean
   restart are covered by regression tests; complete answer-first lines do not
   trigger an unnecessary recovery call.

## Reproduction

```bash
python evaluate_web_hard_stress.py --output_json sample_outputs/web_hard_stress_route.json --output_md sample_outputs/web_hard_stress_route.md
python evaluate_web_hard_stress.py --run-agent --mock --limit 5
python evaluate_web_hard_stress.py --run-agent
python -m pytest -q
```
