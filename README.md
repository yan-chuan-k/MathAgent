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

## Submission Info

Repository URL: `https://github.com/yan-chuan-k/MathAgent`
Branch: `main`
Commit hash: use the final submitted commit SHA.
