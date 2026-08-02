# MathAgent

Challenge: 基于 Intern-S 系列模型的数学智能体设计与推理创新  
Official entry file: `user_agent.py`

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem, metadata)
```

`result` is a JSON-serializable `dict` with a non-empty `final_response` and optional `trace`.

## Install

```bash
pip install -r requirements.txt
```

## Local API Configuration

Only local debugging reads environment variables:

```bash
export INTERN_API_KEY="your_api_key"
export INTERN_MODEL="intern-s2-preview-397b"
export INTERN_API_BASE="https://chat.intern-ai.org.cn/api/v1/"
export LOCAL_MAX_CONCURRENCY=4
```

`user_agent.py` uses the injected official client and does not require local API keys.

## Local Runner

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs --mock
```

For real local Intern-S calls, omit `--mock` after configuring `INTERN_API_KEY`.

Thinking mode is enabled by default for clients that support `thinking_mode`; use `--no-thinking-mode` only for debugging incompatible clients.

## Hard Diagnostics

The repository includes one nontrivial diagnostic problem for each routed domain:

```bash
python diagnose_hard_cases.py --output_file sample_outputs/hard_route_summary.json
python diagnose_hard_cases.py --mock --run-agent --output_file sample_outputs/hard_mock_summary.json
python main.py --input_file sample_data/hard_diagnostics.jsonl --output_dir sample_outputs_hard_mock --mock
```

Without `INTERN_API_KEY`, these commands validate routing and output structure only. Real answer correctness requires running without `--mock` using a configured Intern-S API key.

Current offline diagnostic result:

```text
18 / 18 hard diagnostic cases routed to the expected domain.
18 / 18 hard diagnostic cases completed through the mock ReasoningAgent pipeline.
```

Current real diagnostic notes with local `.env`:

```text
18 / 18 hard diagnostic cases produced non-empty JSON-serializable final_response values.
The differential-geometry case exposed a model formatting issue where K=1 appeared in verification but not final_response.
ReasoningAgent now repairs this specific missing Gaussian-curvature value from structured verification evidence.
```

## Benchmark Distribution Strategy

The router uses the observed 112-problem distribution as a tie-breaking prior. The solver prompt does not see the full distribution; it receives only the routed domain hint.

| Domain | Count | Share |
| --- | ---: | ---: |
| 离散数学 | 24 | 21.43% |
| 数值分析 | 13 | 11.61% |
| 测度积分 | 11 | 9.82% |
| 微分几何 | 9 | 8.04% |
| 概率论 | 8 | 7.14% |
| 抽象代数 | 8 | 7.14% |
| 随机过程 | 7 | 6.25% |
| 复分析 | 7 | 6.25% |
| 常微分方程 | 5 | 4.46% |
| 统计推断 | 4 | 3.57% |
| 泛函分析 | 4 | 3.57% |
| 线性回归 | 3 | 2.68% |
| 偏微分方程 | 3 | 2.68% |
| 非基础及进阶课程 | 2 | 1.79% |
| 高等代数 | 1 | 0.89% |
| 运筹学 | 1 | 0.89% |
| 数学分析 | 1 | 0.89% |
| 拓扑学 | 1 | 0.89% |

Implementation notes:

- `math_agent_core/router.py` contains UTF-8 Chinese and English keywords plus benchmark priors.
- `math_agent_core/prompts.py` builds a focused subject guide from at most three routed domains and isolates the problem statement as untrusted input.
- `math_agent_core/answer_utils.py` normalizes Chinese/English final-answer prefixes, JSON outputs, and LaTeX `\boxed{...}` answers.
- `result_schema.json` and `math_agent_core/schema.py` distinguish schema validity from solve status with system-generated `_meta.overall_status`, `content_complete`, `answer_verified`, and `proof_verified` fields.
- `math_agent_core/tools/sympy_tool.py` provides a safe whitelist SymPy verification layer for arithmetic, equation substitution, expression equivalence, derivative checks, and integral checks.
- `math_agent_core/orchestrator.py` feeds concrete verification failures, residuals, and schema errors into targeted repair prompts instead of retrying the same solver prompt.
- `math_agent_core/candidate.py`, `math_agent_core/search/`, and `math_agent_core/evaluation/` implement candidate snapshots, domain strategy pools, candidate ranking, and answer equivalence clustering.
- `math_agent_core/prompts.py` now separates Solver, Planner, Critic, Reviser, and Finalizer prompt roles. The official path keeps a conservative default budget.
- `math_agent_core/state.py` includes a compact `SolveState`; `math_agent_core/memory/lemma_store.py` tracks open and verified lemmas for future multi-round proof work.
- `math_agent_core/verifiers/completeness.py` adds target coverage and proof-body completeness evidence for multi-part questions and proof tasks.
- `math_agent_core/acceptance.py` centralizes answer acceptance. Evidence now carries `verification_level` and `is_decisive`, so exact symbolic evidence, numeric evidence, model critic opinions, and completeness checks are not treated as equivalent.
- `math_agent_core/tools/matrix_tool.py` and `math_agent_core/verifiers/linear_algebra.py` provide decisive exact checks for determinant, matrix products, inverses, linear-system residuals, ranks, eigenpairs, orthogonality, normalization, and matrix/vector equivalence.
- `user_agent.py` no longer recovers final answers from unverified raw model output. Only system-accepted solved candidates can become `final_response`; otherwise it returns the conservative fallback with trace evidence.
- `main.py` reuses the local API client, skips only parseable successful outputs with non-empty `final_response`, and writes result files atomically for safer resume.

## Validation

Recommended local checks:

```bash
python -c "from user_agent import ReasoningAgent; print('import ok')"
python -m pytest -q
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs --mock
python -m compileall -q .
```

## Submission Info

Repository URL: `https://github.com/yan-chuan-k/MathAgent`  
Branch: `main`  
Commit hash: use the final submitted commit SHA.

## Change Record

### 2026-08-02 Phase 4B Linear Algebra

- Added `MathTool` base and `ToolRegistry` for whitelist tool execution.
- Added `MatrixTool` with bounded exact SymPy checks for determinant, matrix multiplication, inverse verification, linear-system residuals, rank, eigenpair residuals, vector orthogonality, vector normalization, and matrix/vector equivalence.
- Added `math_agent_core/verifiers/linear_algebra.py`, using only structured `requested_checks` instead of natural-language code execution.
- Extended `requested_checks` schema and normalization to preserve nested matrix/vector arguments safely.
- Integrated linear algebra evidence into orchestrator verification; decisive matrix failures reject candidates and decisive matrix passes can satisfy `AcceptancePolicy`.
- Added tests for determinant pass, wrong eigenpair rejection, requested-check dispatch, and orchestrator acceptance with matrix evidence.
- Validation: `python -m pytest -q` -> 47 passed, 1 skipped; `python -m compileall -q .`; mock baseline runner passed.

### 2026-08-02 Phase 4A

- Added `VerificationLevel` and extended `VerificationEvidence` with `verification_level` and `is_decisive`.
- Added `math_agent_core/acceptance.py` with `AcceptancePolicy`, making `solved/probable/uncertain/invalid` decisions independent of model self-confidence.
- Marked SymPy exact checks as decisive `exact_symbolic` evidence; completeness checks as `completeness_only`; Critic reviews as non-decisive `model_critic`.
- Enforced evidence priority: decisive tool failures reject candidates even if Critic passes; Critic-only and completeness-only support cannot produce `solved`; high-precision numeric support is capped at `probable`.
- Added tests for exact symbolic acceptance, decisive tool failure over model pass, numeric-only probable status, completeness-only uncertainty, and model-critic-only probable status.
- Validation: `python -m pytest -q` -> 43 passed, 1 skipped; `python -m compileall -q .`; mock baseline runner passed.

### 2026-08-02 Phase 3

- Added compact `SolveState` snapshots for open goals, rejected attempts, rejected strategies, verification evidence, and budget.
- Added `LemmaStore` and `Lemma` data structures to track open/verified lemmas and candidate usage without retaining full conversations.
- Added completeness verifier for answer target coverage and proof-body checks; missing multi-part targets now generate concrete `missing_case` evidence.
- Integrated completeness evidence into candidate assessment and trace state, while ensuring completeness-only pass cannot mark an answer as mathematically verified.
- Added tests for target extraction, missing target rejection, short proof rejection, LemmaStore, SolveState compacting, and orchestrator open-goal tracking.
- Validation: `python -m pytest -q` -> 38 passed, 1 skipped; `python -m compileall -q .`; mock baseline runner passed.

### 2026-08-02 Phase 2

- Added strategy-aware candidate search with `CandidateSolution`, domain strategy pools, ranker scoring, and answer equivalence clustering.
- Split prompt roles into Planner, Solver, Critic, Reviser, and Finalizer builders while preserving the existing solver schema contract.
- Updated `MathAgentOrchestrator` to generate candidates by strategy, run tool verification and optional Critic review per candidate, cluster/rank candidates, and select the highest scoring verified candidate.
- Added optional Finalizer formatting that can only compress the system-selected candidate into `final_response`; official `ReasoningAgent` keeps it disabled by default to avoid extra API cost.
- Extended `ScriptedClient` with role-aware response queues for offline multi-agent tests.
- Added tests for multi-candidate ranking, Critic rejection, Finalizer formatting, and role-separated scripted responses.
- Validation: `python -m pytest -q` -> 32 passed, 1 skipped; `python -m compileall -q .`; mock baseline runner passed.

### 2026-08-02

- Added first-stage correctness architecture: system-owned solve status, verification evidence, and strict separation between `schema_valid` and mathematical acceptance.
- Added `VerificationEvidence`, `SolveAssessment`, `OverallStatus`, and `FailureKind` in `math_agent_core/state.py`.
- Added safe SymPy whitelist checks in `math_agent_core/tools/sympy_tool.py`; no model-generated Python, shell, file, or network code is executed.
- Changed orchestrator retries to targeted repair prompts containing schema errors, failure kind, residuals, previous answer, and verifier evidence.
- Ignored all model-provided `_meta` fields during normalization; `_meta.model`, `backend`, `attempts`, `elapsed_seconds`, `schema_valid`, and solve-status fields are generated by code only.
- Removed unverified raw-output answer recovery from `ReasoningAgent` and enabled tool verification by default for the official path.
- Added `ScriptedClient` and `FaultInjectionClient` for offline tests of invalid JSON, wrong-answer rejection, repair prompts, and `_meta` forgery.
- Hardened `main.py` resume behavior and atomic writes; failed or empty output files are no longer skipped.
- Added API retry jitter and best-effort `Retry-After` handling in `intern_s1_client.py` without logging secrets.
- Validation: `python -c "from user_agent import ReasoningAgent; print('import ok')"`, `python -m pytest -q` -> 29 passed, 1 skipped, `python -m compileall -q .`, and mock baseline runner passed.

### 2026-08-01

- Added hard diagnostic problems covering 18 routed domains and a `diagnose_hard_cases.py` script.
- Confirmed `.env` loading for real local Intern-S diagnostics without printing secrets.
- Fixed Windows console Unicode printing in `diagnose_hard_cases.py`.
- Corrected hard diagnostic answer hints for measure integration, complex analysis, and ODE.
- Added a conservative final-response repair for Gaussian-curvature tasks when verification contains `K = ... = value` but final answer omits it.
- Verified offline routing on the hard diagnostics: 18/18 route hits; mock ReasoningAgent pipeline completed 18/18.
- Refactored solver prompt to remove benchmark priors from model-visible context; priors remain only in router scoring.
- Limited focused domain guidance to at most three routed domains and added input-payload isolation against prompt injection.
- Changed the model-facing verification contract from free-form `verification_process` to concrete `verification.checks`, while preserving internal compatibility.
- Expanded schema compatibility for construction, counterexample, classification, vector, function, distribution, choice, boolean, and text answers.
- Added the 112-problem benchmark distribution as router priors.
- Rebuilt route keywords for high-frequency domains: discrete math, numerical analysis, measure integration, differential geometry, probability, abstract algebra, stochastic process, and complex analysis.
- Split probability, stochastic process, statistics, and linear regression into separate routing domains.
- Replaced mojibake prompt/answer/README text with valid UTF-8 Chinese.
- Added focused domain guides and final-answer formatting hints in the solver prompt.
- Expanded tests for router distribution priors, stochastic process, linear regression, PDE, prompt contract, schema contract, and Chinese answer extraction.
- Updated `math_agent_project_record.docx` via `update_project_record.py`; if the original Word file is open and locked, the script writes `math_agent_project_record_updated.docx` as the synchronized copy.
