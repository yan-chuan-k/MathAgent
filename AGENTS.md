# AGENTS.md

> 本文件用于指导 Codex / 代码智能体继续开发本项目。  
> 本版已按《初赛赛题介绍与提交要求》进行更新，优先满足挑战杯初赛平台的真实评测接口。  
> 核心目标：让提交仓库在官方 runner 中可以被正常安装、import、初始化、调用，并返回可判分的 `final_response`。

---

## 0. 最高优先级：官方初赛接口规范

本项目的最终评测入口不是 `main.py`，也不是 `run_batch.py`，而是仓库根目录下的：

```text
user_agent.py
```

官方平台会执行类似逻辑：

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem, metadata)
```

因此 Codex 的第一任务是保证：

```text
仓库根目录存在 user_agent.py
user_agent.py 中存在 ReasoningAgent 类
ReasoningAgent 能被 official_client 初始化
ReasoningAgent.solve(problem: str, metadata: dict) 能被调用
solve 返回 dict
dict 中必须包含非空字符串 final_response
返回值必须 JSON 可序列化
```

所有其它工程结构、Lagent、多智能体、工具校验、日志系统、Demo、批处理脚本，都只能作为辅助能力，不能破坏上述接口。

---

## 1. 官方评测任务理解

赛题要求实现：

```python
agent.solve(problem: str, metadata: dict) -> dict
```

输入：

```text
problem  : 数学题题面文本
metadata : 题目元信息字典，正式评测时至少可能包含 idx，具体字段以官方 runner 为准
```

输出必须是：

```json
{
  "final_response": "最终答案"
}
```

推荐同时返回：

```json
{
  "final_response": "最终答案",
  "trace": [
    {"step": "plan", "content": "..."},
    {"step": "model_call", "content": "..."},
    {"step": "finalize", "content": "..."}
  ]
}
```

官方主要根据 `final_response` 进行判分。`trace` 主要用于异常排查、展示和同分情况下的设计质量参考。因此：

```text
final_response 要短、准、可判分
trace 可以记录推理过程，但不能替代 final_response
```

---

## 2. 必须遵守的文件要求

仓库根目录必须包含：

```text
user_agent.py
```

建议同时包含：

```text
requirements.txt
README.md
main.py
sample_data/
tools/
prompts/
utils/
math_agent_core/
```

但是正式评测最关键的是：

```text
user_agent.py 可以 import
ReasoningAgent 可以初始化
solve 可以返回合法 dict
requirements.txt 可以安装成功
```

不要依赖用户本机上的绝对路径，例如：

```text
D:\PyCharm\...
/home/xxx/...
C:\Users\xxx\...
```

所有文件读取必须使用相对路径或基于 `Path(__file__).resolve().parent`。

---

## 3. user_agent.py 必须实现的内容

`user_agent.py` 必须提供：

```python
class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        ...

    def solve(self, problem: str, metadata: dict) -> dict:
        ...
```

### 3.1 构造函数要求

官方会用：

```python
ReasoningAgent(client=official_client)
```

初始化，因此构造函数必须兼容：

```python
def __init__(self, client, *args, **kwargs):
    self.client = client
```

注意：

1. `client` 由平台提供。
2. 不要在 `ReasoningAgent` 中强制读取 `.env`。
3. 不要在 `ReasoningAgent` 中写死 API key。
4. 不要假设本地存在固定 API 配置文件。
5. 不要在构造函数中做耗时网络请求。
6. 不要在构造函数中批量加载巨大文件。
7. 不要因为某个可选依赖缺失导致初始化失败。

### 3.2 solve 函数要求

`solve` 必须满足：

```python
def solve(self, problem: str, metadata: dict) -> dict:
    ...
```

要求：

1. `problem` 是题目文本。
2. `metadata` 是元信息字典。
3. 不能依赖 `metadata["answer"]`。
4. 不能读取标准答案、隐藏测试集或 judger 信息。
5. 单题异常不能直接抛出到平台。
6. 即使内部失败，也尽量返回一个 JSON 可序列化 dict。
7. `final_response` 必须是非空字符串。
8. 不要返回自定义对象、Path、set、bytes、异常对象等不可 JSON 序列化内容。

推荐失败兜底：

```python
return {
    "final_response": "无法确定",
    "trace": [
        {"step": "error", "content": "内部推理失败，已返回兜底答案。"}
    ]
}
```

但是正式冲分时不应频繁兜底，应尽量返回模型推理后的最佳答案。

---

## 4. 官方 client 使用规范

正式评测时，平台提供的 `client` 与 baseline 中 `llm_client.py` 的 `InternChatClient` 结构一致，可参考：

```python
response = client.chat(
    messages=[
        {"role": "user", "content": problem}
    ],
    temperature=0.2,
    max_tokens=4096,
)
```

Codex 必须让项目内部模型调用走：

```python
self.client.chat(messages=..., temperature=..., max_tokens=...)
```

而不是强制使用自己写死的 OpenAI client。

### 4.1 official path 与 local path 分离

必须区分：

```text
official path : user_agent.py 使用官方传入 client
local path    : main.py / 本地调试脚本可以使用本地 InternChatClient 和环境变量
```

不允许在 `user_agent.py` 的正式路径中强制读取：

```text
INTERN_API_KEY
INTERN_API_BASE
INTERN_MODEL
.env
```

但本地调试代码可以支持：

```text
export INTERN_API_KEY="sk-..."
export INTERN_MODEL="intern-s2-preview"
export INTERN_API_BASE="..."
```

### 4.2 模型选择

本地可用模型包括但不限于：

```text
intern-s1
intern-s1-pro
intern-s2-preview
```

正式提交时，选手可以在判分系统中选择模型。代码不应写死只能用某一个模型。

---

## 5. final_response 设计原则

`final_response` 是最重要字段。

### 5.1 计算题

应尽量输出简洁答案：

```text
72
```

```text
x = 1/2
```

```text
最大值 2，最小值 -2
```

不要输出长篇推理：

```text
首先我们观察到...所以答案是 72
```

长推理放到 `trace`，不是 `final_response`。

### 5.2 证明题

如果题目要求证明，`final_response` 可以是一段完整但尽量简洁的证明。要求：

1. 逻辑闭合。
2. 不要只写“成立”。
3. 不要只写最终结论。
4. 不要过度冗长。
5. 关键定理条件要写清楚。

### 5.3 选择题 / 判断题

优先输出：

```text
A
```

或：

```text
正确
```

如果需要解释，放入 `trace`。

### 5.4 表达式题

优先输出标准数学形式：

```text
e^{-\\pi^2 t}\\sin(\\pi x)
```

或者：

```text
u(x,t)=e^{-\\pi^2 t}\\sin(\\pi x)
```

根据题目是否要求函数名决定。

---

## 6. trace 设计原则

`trace` 推荐为列表：

```json
[
  {"step": "parse", "content": "识别为抽象代数有限域问题。"},
  {"step": "plan", "content": "计算 F_81 over F_3 的生成元个数。"},
  {"step": "solve", "content": "F_81 的真子域为 F_3 和 F_9，不能生成全域的元素都在 F_9 中。"},
  {"step": "finalize", "content": "最终答案为 81-9=72。"}
]
```

要求：

1. 必须 JSON 可序列化。
2. 不要放 API key。
3. 不要放访问令牌。
4. 不要放个人隐私。
5. 不要放完整原始隐藏题库。
6. 不要无限增长。
7. 不要放无法序列化的对象。
8. 内容尽量概括，避免 token 过大。

建议 trace 记录：

```text
parse
route
plan
candidate
verify
repair
finalize
error
```

不建议记录：

```text
完整大模型原始长输出
敏感请求头
API Key
本地绝对路径
标准答案
judger 细节
```

---

## 7. 当前项目应如何迁移

当前项目已有：

```text
intern_s1_client.py
math_agent.py
main.py
run_batch.py
result_schema.json
json_validator.py
result_validator.py
validate_results.py
problems.jsonl
results.jsonl
logs/
```

Codex 不要直接删除它们。应采用兼容迁移：

```text
第一层：user_agent.py             # 官方入口，必须稳定
第二层：math_agent_core/          # 内部推理流程
第三层：main.py / run_batch.py    # 本地调试入口
第四层：logs / outputs            # 本地观察，不影响官方评测
```

推荐新增：

```text
user_agent.py
math_agent_core/
├─ __init__.py
├─ orchestrator.py
├─ prompts.py
├─ answer_utils.py
├─ json_utils.py
├─ trace_utils.py
├─ router.py
├─ verifier.py
├─ repair.py
└─ tools/
   ├─ __init__.py
   ├─ symbolic_checker.py
   ├─ matrix_checker.py
   └─ numeric_checker.py
```

---

## 8. user_agent.py 推荐骨架

Codex 可以按下面骨架实现，但要结合当前项目实际文件名调整：

```python
from __future__ import annotations

import traceback
from typing import Any, Dict, List


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.max_retries = int(kwargs.get("max_retries", 1))
        self.temperature = float(kwargs.get("temperature", 0.2))
        self.max_tokens = int(kwargs.get("max_tokens", 4096))

        try:
            from math_agent_core.orchestrator import MathAgentOrchestrator
            self.orchestrator = MathAgentOrchestrator(
                client=self.client,
                max_retries=self.max_retries,
            )
        except Exception:
            self.orchestrator = None

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, str]] = []

        try:
            if not isinstance(problem, str) or not problem.strip():
                return {
                    "final_response": "无法确定",
                    "trace": [{"step": "error", "content": "题面为空或不是字符串。"}],
                }

            safe_metadata = metadata if isinstance(metadata, dict) else {}
            idx = safe_metadata.get("idx", safe_metadata.get("id", ""))

            if self.orchestrator is not None:
                result = self.orchestrator.solve(problem=problem, metadata=safe_metadata)
                final_response = self._extract_final_response(result)
                trace = self._extract_trace(result)
                return {
                    "final_response": final_response,
                    "trace": trace,
                }

            # fallback: direct model call
            messages = self._build_direct_prompt(problem, safe_metadata)
            response = self.client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            final_response = self._normalize_model_response(response)
            return {
                "final_response": final_response or "无法确定",
                "trace": [
                    {"step": "fallback", "content": "使用直接模型调用生成答案。"}
                ],
            }

        except Exception as exc:
            return {
                "final_response": "无法确定",
                "trace": [
                    {"step": "error", "content": f"{type(exc).__name__}: {str(exc)[:300]}"}
                ],
            }

    def _build_direct_prompt(self, problem: str, metadata: Dict[str, Any]):
        return [
            {
                "role": "system",
                "content": (
                    "你是严谨的数学解题智能体。请解答题目，并只在最后给出简洁的最终答案。"
                    "如果是计算题，最终答案尽量短；如果是证明题，给出简洁完整证明。"
                ),
            },
            {
                "role": "user",
                "content": f"题目：{problem}\n请给出最终答案。",
            },
        ]

    def _normalize_model_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            for key in ("final_response", "content", "text", "answer"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(response).strip()

    def _extract_final_response(self, result: Any) -> str:
        if isinstance(result, dict):
            value = result.get("final_response")
            if isinstance(value, str) and value.strip():
                return value.strip()
            final_answer = result.get("final_answer")
            if isinstance(final_answer, dict):
                answer = final_answer.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
            if isinstance(final_answer, str) and final_answer.strip():
                return final_answer.strip()
        return "无法确定"

    def _extract_trace(self, result: Any):
        if isinstance(result, dict) and isinstance(result.get("trace"), list):
            return result["trace"]
        if isinstance(result, dict):
            trace = []
            if result.get("reasoning_plan"):
                trace.append({"step": "plan", "content": str(result.get("reasoning_plan"))[:1000]})
            if result.get("verification"):
                trace.append({"step": "verify", "content": str(result.get("verification"))[:1000]})
            return trace
        return []
```

这个骨架的关键点：

1. 官方 client 从外部注入。
2. 不依赖本地 API key。
3. 内部 orchestrator 失败时有 fallback。
4. solve 不向外抛异常。
5. 返回 dict 且含非空 `final_response`。
6. trace 可选但安全。

Codex 可优化，但不得破坏这些性质。

---

## 9. 本地调试输入输出规范

官方 baseline 本地调试输入为 JSONL，每行一道题，至少包含：

```json
{"idx": 0, "problem": "题目文本"}
```

样例可能包含：

```json
{
  "idx": 0,
  "problem": "设$\\mathbb{F}_{81}$为$81$元的有限域...",
  "answer": "72",
  "subject": "抽象代数",
  "source": "sample"
}
```

注意：

```text
answer 只出现在样例数据中用于本地对照
正式评测不会向 solve 传入标准答案
代码不得依赖 answer 字段
```

本地 runner 输出为：

```text
outputs/
├─ 0.json
├─ 1.json
└─ 2.json
```

成功样例：

```json
{
  "idx": 0,
  "status": "success",
  "final_response": "72",
  "trace": [
    {
      "step": "solve",
      "content": "有限域 F_81 是 F_3 上 4 维扩张，生成整个扩张的元素个数为 72。"
    }
  ]
}
```

异常样例：

```json
{
  "idx": 0,
  "status": "error",
  "final_response": "",
  "error": {
    "type": "RuntimeError",
    "message": "错误信息"
  },
  "trace": []
}
```

如果某个 `idx.json` 已经存在且文件非空，runner 会跳过该题，便于中断后继续运行。

---

## 10. main.py 必须兼容官方 baseline 调试命令

官方建议本地调试命令：

```bash
pip install -r requirements.txt
export INTERN_API_KEY="sk-..."
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

因此 Codex 必须让 `main.py` 支持：

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

也可以额外兼容旧命令：

```bash
python main.py --input input.json --output result.json
python run_batch.py --input problems.jsonl --output results.jsonl
```

但不能只支持旧命令而不支持官方 baseline 命令。

推荐 `main.py` 职责：

1. 读取 JSONL。
2. 初始化本地 client。
3. 初始化 `ReasoningAgent(client=client)`。
4. 对每行题目调用 `agent.solve(problem, metadata)`。
5. 保存到 `output_dir/{idx}.json`。
6. 已存在非空文件则跳过。
7. 支持并发数环境变量：
   ```text
   LOCAL_MAX_CONCURRENCY
   ```
8. 默认并发数可设为 8，但要允许用户调小。

---

## 11. 本地 client 与环境变量

本地调试可以使用环境变量：

```bash
export INTERN_API_KEY="sk-..."
export INTERN_MODEL="intern-s2-preview"
export INTERN_API_BASE="..."
export LOCAL_MAX_CONCURRENCY=4
```

注意：

1. `INTERN_API_KEY` 只用于本地调试。
2. `user_agent.py` 的正式路径不应要求该变量存在。
3. `.env` 不能提交到仓库。
4. 日志中不能出现 API key。
5. 运行报错时不能打印完整请求头。

推荐本地 client 文件：

```text
llm_client.py
```

或复用 baseline 的 `InternChatClient`。

---

## 12. 推理流程建议

官方允许探索：

```text
提示词设计与多轮推理
多候选生成、验证与选择
规划、反思、纠错、答案格式化
工具调用、检索、记忆或其它推理策略
面向数学题的专门解析、符号计算或后处理
```

但在初赛中，稳定性优先。推荐默认流程：

```text
题面 problem + metadata
  ↓
题型快速识别 route
  ↓
解题策略 planning
  ↓
模型生成候选答案 candidate
  ↓
答案抽取 answer extraction
  ↓
轻量校验 verify
  ↓
必要时一次修复 repair
  ↓
格式化 final_response
  ↓
返回 trace
```

不要默认进行过多模型调用。正式评测有超时、并发、token 预算与资源限制。

推荐调用次数：

```text
普通题：1 次模型调用
中等题：1 次解题 + 1 次校验/修复
难题：最多 3 个候选 + 1 次选择
```

默认 `max_retries` 不超过 1 或 2。

---

## 13. Prompt 设计要求

核心 prompt 应服务于 `final_response` 的可判分性。

### 13.1 直接求解 prompt

```text
你是一个严谨的数学解题智能体。
请解答给定数学题。
要求：
1. 先在内部完成推理。
2. 最终答案必须明确、简洁、可判分。
3. 计算题 final answer 尽量只给答案。
4. 证明题给出简洁完整证明。
5. 不要编造题目没有给出的条件。
```

### 13.2 答案抽取 prompt

当模型返回过长时，可再次调用或本地抽取：

```text
请从下面的解题过程里抽取最终答案。
只输出最终答案，不要解释。
```

### 13.3 校验 prompt

```text
你是数学答案审计智能体。
请检查候选答案是否真正回答题目，是否有计算错误、逻辑跳步、格式不清。
如果答案可用，请给出简短 final_response。
如果不可用，请说明错误并给出修正答案。
```

---

## 14. 答案抽取与格式化

Codex 必须实现本地函数：

```python
extract_final_answer(text: str, problem: str | None = None) -> str
normalize_final_response(answer: str) -> str
```

处理：

1. 删除 Markdown 代码块。
2. 删除“最终答案：”前缀但保留答案本身。
3. 删除多余客套话。
4. 对明显 JSON 输出提取 `final_response` 或 `answer`。
5. 对过长文本保留最后明确答案段。
6. 对空字符串返回兜底 `"无法确定"`。
7. 限制长度，避免 `final_response` 过长。

建议长度：

```text
普通计算题 final_response 不超过 500 字符
证明题可放宽到 3000 字符
trace 单条 content 不超过 1000 字符
```

---

## 15. 题型路由建议

Router 可根据题面和 metadata 中的 `subject`、`type`、`category` 辅助判断，但不能依赖它们一定存在。

至少识别：

```text
calculus
real_analysis
linear_algebra
abstract_algebra
complex_analysis
ode
pde
probability
statistics
optimization
topology
number_theory
combinatorics
geometry
functional_analysis
measure_theory
discrete_math
unknown
```

不同题型可以使用不同 prompt，但输出最终都要归一到：

```python
{"final_response": "...", "trace": [...]}
```

---

## 16. 工具校验建议

可以使用轻量工具提高正确率，但不得让工具成为评测风险来源。

推荐依赖：

```text
sympy
numpy
```

推荐校验：

```text
矩阵行列式、秩、特征值
简单导数、积分、极值
方程解代回
ODE/PDE 简单残差
概率分布求和
优化约束检查
```

要求：

1. 工具校验失败不能导致 `solve` 抛异常。
2. 工具只能作为辅助，不要执行危险代码。
3. 不要让模型生成任意 Python 后直接执行。
4. 工具调用必须有 try/except。
5. 校验结果可以写入 trace，但不要太长。

---

## 17. Lagent 使用要求

可以使用 baseline 中的 Lagent 示例，也可以不用 Lagent。官方只要求 `user_agent.py` 暴露符合规范的 `ReasoningAgent` 和 `solve` 方法。

因此：

```text
Lagent 是可选加分项，不是必需项
```

Codex 若接入 Lagent，必须满足：

1. 不影响 `user_agent.py` 的官方入口。
2. 不让官方评测必须安装复杂不可控依赖。
3. 如果 Lagent import 失败，应回退到 simple pipeline。
4. 不要使用 Lagent 执行危险系统命令。
5. 不要因为 Lagent 状态管理导致题目之间相互污染。

推荐：

```text
backend="simple" 作为默认
backend="lagent" 作为可选
```

官方初赛提交建议默认 simple，决赛展示再强化 lagent 多智能体。

---

## 18. 不允许事项

Codex 必须避免：

1. 硬编码 API key。
2. 硬编码样例题答案。
3. 读取或构造隐藏测试集。
4. 依赖标准答案 `answer` 字段。
5. 依赖本地绝对路径。
6. 依赖题目固定顺序。
7. 假设多个题目一定在同一进程中运行。
8. 在 trace 中写入密钥、令牌、个人隐私。
9. 输出恶意内容。
10. 执行破坏性操作。
11. 规避平台资源限制。
12. 在 import `user_agent.py` 时就调用模型。
13. 在构造函数里做大量耗时操作。
14. 返回不可 JSON 序列化对象。
15. 让 `final_response` 为空。

---

## 19. 异常处理要求

官方常见异常包括：

```text
user_agent.py 不存在
ReasoningAgent 类不存在
构造函数不接受 client
solve 方法不存在
solve 签名不符合要求
solve 抛出异常
返回值不是字典
final_response 缺失
final_response 为空
final_response 不是字符串
返回值无法 JSON 序列化
依赖无法安装
运行超时
资源使用超过限制
依赖本地绝对路径
代码包含硬编码密钥
```

Codex 必须针对这些异常做防护。

`solve` 内部结构建议：

```python
try:
    ...
except Exception as exc:
    return {
        "final_response": "无法确定",
        "trace": [
            {"step": "error", "content": f"{type(exc).__name__}: {str(exc)[:300]}"}
        ],
    }
```

---

## 20. requirements.txt 要求

所有依赖必须写入 `requirements.txt`。

建议轻量化：

```text
openai>=1.0.0
requests>=2.31.0
python-dotenv>=1.0.0
sympy>=1.12
numpy>=1.24
tqdm>=4.66
```

如果使用 `jsonschema`：

```text
jsonschema>=4.0.0
```

如果使用 Lagent，要确保安装方式在 README 中写清楚，但不要让基础入口必须依赖 Lagent 才能 import。

不要加入大型 GPU 依赖，除非确有必要。

---

## 21. README 必须说明

README 至少写明：

```text
项目名称
赛题名称
入口文件 user_agent.py
ReasoningAgent 使用方式
依赖安装方式
本地 API key 配置方式
本地调试命令
选择使用的模型
仓库地址
分支名称
commit hash
异常说明
```

API key 只能写成：

```text
export INTERN_API_KEY="your_api_key"
```

不能写真实 key。

---

## 22. 提交方式要求

官方提交方式包括：

1. 判分系统提交：
   ```text
   仓库地址 + commit SHA
   ```
2. 初赛截止前按官网要求将最终版本代码仓库与其它材料打包成 `.zip` 发送邮件。

建议压缩包至少包含：

```text
最终版本代码
user_agent.py
requirements.txt
README 或说明文件
队伍信息
题目名称
仓库地址
分支名称
commit hash
选择使用的模型
```

邮件评测模板中需要填写：

```text
队伍名称
题目名称：基于 Intern-S 系列模型的数学智能体设计与推理创新
报名成员信息
仓库地址
分支名称
commit hash
选择使用的模型
代码库 zip 文件名
备注
```

注意：

```text
提交的是 commit SHA，不是只提交分支名
附件代码版本要与邮件正文 commit hash 对应
```

---

## 23. 提交次数与本地调试

判分系统上线后，每支队伍提交次数有限：

```text
每天最多 2 次
每周最多 10 次
```

因此 Codex 必须优先完善本地调试，避免浪费提交机会。

本地至少跑通：

```bash
pip install -r requirements.txt
python -c "from user_agent import ReasoningAgent; print(ReasoningAgent)"
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

如果有 mock client，至少跑通：

```bash
python tests/test_user_agent_entry.py
```

---

## 24. 推荐测试文件

新增：

```text
tests/
├─ test_user_agent_entry.py
├─ test_json_serializable.py
├─ test_no_secret_leak.py
├─ test_answer_extraction.py
└─ test_sample_runner.py
```

### 24.1 test_user_agent_entry.py

必须检查：

```python
from user_agent import ReasoningAgent

class MockClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        return "72"

agent = ReasoningAgent(client=MockClient())
result = agent.solve("1+1=?", {"idx": 0})

assert isinstance(result, dict)
assert isinstance(result.get("final_response"), str)
assert result["final_response"].strip()
```

### 24.2 test_json_serializable.py

检查：

```python
json.dumps(result, ensure_ascii=False)
```

### 24.3 test_no_secret_leak.py

检查输出中不包含：

```text
sk-
INTERN_API_KEY
api_key
```

---

## 25. 本地 runner 建议实现

如果当前仓库没有官方 baseline runner，应在 `main.py` 中实现最小 runner：

```python
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from user_agent import ReasoningAgent
from llm_client import InternChatClient


def load_jsonl(path):
    ...


def run_one(agent, item):
    idx = item.get("idx")
    problem = item.get("problem")
    metadata = {k: v for k, v in item.items() if k != "problem"}
    result = agent.solve(problem, metadata)
    return {
        "idx": idx,
        "status": "success",
        "final_response": result.get("final_response", ""),
        "trace": result.get("trace", []),
    }
```

要求：

1. 输出 `output_dir/{idx}.json`。
2. 已存在非空则跳过。
3. 每题异常写 error JSON。
4. 并发数读取 `LOCAL_MAX_CONCURRENCY`。
5. 不把 `answer` 传给用于推理的 prompt。
6. 本地可以保留 `answer` 做离线对比，但绝不能在 solve 中使用。

---

## 26. Codex 开发顺序

### Phase 1：补齐官方入口

目标：

```text
user_agent.py 存在且符合官方接口
```

任务：

1. 新建/修改 `user_agent.py`。
2. 实现 `ReasoningAgent(client, *args, **kwargs)`。
3. 实现 `solve(problem, metadata)`。
4. 返回 `final_response` 和 `trace`。
5. 加 mock 测试。

验收：

```bash
python -c "from user_agent import ReasoningAgent; print('ok')"
```

### Phase 2：打通本地 baseline runner

目标：

```text
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

任务：

1. 兼容 `--input_file`。
2. 兼容 `--output_dir`。
3. 输出 `{idx}.json`。
4. 支持跳过已有结果。
5. 支持 `LOCAL_MAX_CONCURRENCY`。

### Phase 3：接入当前 MathAgentOrchestrator

目标：

```text
user_agent.py 调用内部模块，而不是只有直接 prompt
```

任务：

1. 将现有 `math_agent.py` 封装为 orchestrator。
2. 适配 client.chat。
3. 输出转成 `final_response`。
4. trace 记录关键步骤。

### Phase 4：答案抽取与格式化

目标：

```text
final_response 更短、更准、更适合 judger
```

任务：

1. 实现 `extract_final_answer`。
2. 实现 `normalize_final_response`。
3. 针对计算题、证明题、选择题做差异处理。

### Phase 5：轻量校验与修复

目标：

```text
减少明显错答和格式错误
```

任务：

1. 加 SymPy / NumPy 工具。
2. 加 Verifier。
3. 加一次 Repair。
4. 控制 token 和超时风险。

### Phase 6：Lagent 可选增强

目标：

```text
不影响初赛入口的前提下，保留多智能体扩展
```

任务：

1. 将 Lagent 放入可选 backend。
2. import 失败自动回退。
3. 不作为默认依赖阻塞评测。

---

## 27. 最小验收命令

Codex 每轮修改后至少运行：

```bash
python -c "from user_agent import ReasoningAgent; print('import ok')"
```

```bash
python - <<'PY'
from user_agent import ReasoningAgent
import json

class MockClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        return "2"

agent = ReasoningAgent(client=MockClient())
result = agent.solve("1+1=?", {"idx": 0})
assert isinstance(result, dict)
assert isinstance(result.get("final_response"), str)
assert result["final_response"].strip()
json.dumps(result, ensure_ascii=False)
print(result)
PY
```

如果有样例数据：

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

如果仍保留旧批处理：

```bash
python run_batch.py --input problems.jsonl --output results.jsonl
```

---

## 28. 提交前检查清单

提交 commit 前，Codex 必须确认：

```text
[ ] 仓库根目录存在 user_agent.py
[ ] user_agent.py 可被 import
[ ] 存在 ReasoningAgent 类
[ ] ReasoningAgent(client=official_client) 可初始化
[ ] solve(problem, metadata) 签名正确
[ ] solve 返回 dict
[ ] final_response 存在
[ ] final_response 是非空字符串
[ ] 返回值 JSON 可序列化
[ ] trace 不包含密钥和隐私
[ ] requirements.txt 完整
[ ] 代码不依赖绝对路径
[ ] 代码不依赖 answer 字段
[ ] 代码没有硬编码 API key
[ ] 代码没有硬编码样例题答案
[ ] 本地 main.py baseline 命令可运行
[ ] README 写明运行方式、模型、commit 信息
[ ] 提交的是 commit SHA，不只是分支名
```

---

## 29. 一等奖导向但不破坏官方接口的增强方向

在满足官方接口后，再考虑增强：

1. 多候选生成：
   ```text
   candidate_1, candidate_2, candidate_3 → verifier/judge → final_response
   ```

2. 题型专用 prompt：
   ```text
   抽象代数、复分析、拓扑、PDE、运筹优化等分别设计 prompt
   ```

3. 工具校验：
   ```text
   SymPy/NumPy 检查可计算题
   ```

4. 反思修复：
   ```text
   candidate → audit → repair → final
   ```

5. 答案压缩：
   ```text
   长推理 → 简洁 final_response
   ```

6. trace 质量：
   ```text
   记录 plan、candidate、verify、finalize，方便展示系统设计
   ```

这些增强都必须围绕：

```text
提升 final_response 正确率
```

而不是为了展示复杂度。

---

## 30. 最终一句话原则

本项目的初赛提交目标是：

```text
在官方 runner 中稳定 import user_agent.py，
用官方 client 初始化 ReasoningAgent，
对每道隐藏数学题调用 solve(problem, metadata)，
返回非空、简洁、可判分的 final_response，
并用安全的 trace 展示推理过程。
```

任何代码修改都不能违背这一目标。
