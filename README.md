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

## Benchmark Distribution Strategy

The router and prompt strategy use the observed 112-problem distribution as a tie-breaking prior. The problem statement and metadata remain authoritative.

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
- `math_agent_core/prompts.py` builds a focused subject guide from the top routed domains and includes output hints per domain.
- `math_agent_core/answer_utils.py` normalizes Chinese/English final-answer prefixes, JSON outputs, and LaTeX `\boxed{...}` answers.

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

### 2026-08-01

- Refactored solver prompt to remove benchmark priors from model-visible context; priors remain only in router scoring.
- Limited focused domain guidance to at most three routed domains and added input-payload isolation against prompt injection.
- Changed the model-facing verification contract from free-form `verification_process` to concrete `verification.checks`, while preserving internal compatibility.
- Expanded schema compatibility for construction, counterexample, classification, vector, function, distribution, choice, boolean, and text answers.
- Added the 112-problem benchmark distribution as router priors.
- Rebuilt route keywords for high-frequency domains: discrete math, numerical analysis, measure integration, differential geometry, probability, abstract algebra, stochastic process, and complex analysis.
- Split probability, stochastic process, statistics, and linear regression into separate routing domains.
- Replaced mojibake prompt/answer text with valid UTF-8 Chinese.
- Added focused domain guides and final-answer formatting hints in the solver prompt.
- Expanded tests for router distribution priors, stochastic process, linear regression, PDE, and Chinese answer extraction.
- Updated `math_agent_project_record.docx` via `update_project_record.py`; if the original Word file is open and locked, the script writes `math_agent_project_record_updated.docx` as the synchronized copy.
