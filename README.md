# Math-Verify Agent

Challenge: 基于 Intern-S 系列模型的数学智能体设计与推理创新

Official entry file: `user_agent.py`

The official runner should use:

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

`user_agent.py` does not read `.env` or require local API keys when the official client is injected.

## Local Runner

Official baseline-style JSONL:

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs --mock
```

Legacy single-problem mode:

```bash
python main.py --input input.json --output result.json --mock
python run_batch.py --input problems.jsonl --output results.jsonl --mock
```

## Model

Local default model: `intern-s2-preview-397b`, configurable through `INTERN_MODEL`. The judging platform may inject its own client/model.

## Submission Info

Repository URL: `https://github.com/yan-chuan-k/-.git`
Branch: `main`
Commit hash: use the final submitted commit SHA.

## Error Handling

`ReasoningAgent.solve` catches per-problem failures and returns a JSON-serializable dict with a non-empty `final_response`. The official entry path uses the injected client and does not require local API keys.
