import json

from user_agent import ReasoningAgent


class MockClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        return "2"


class ThinkingMockClient:
    def __init__(self):
        self.thinking_mode_values = []

    def chat(self, messages, temperature=0.2, max_tokens=4096, thinking_mode=None):
        self.thinking_mode_values.append(thinking_mode)
        return '{"final_answer": {"answer": "2"}, "verification": {"verification_result": "pass", "confidence": 0.99}}'


def test_user_agent_entry_returns_final_response():
    agent = ReasoningAgent(client=MockClient())
    result = agent.solve("1+1=?", {"idx": 0})

    assert isinstance(result, dict)
    assert isinstance(result.get("final_response"), str)
    assert result["final_response"].strip()


def test_user_agent_result_is_json_serializable():
    agent = ReasoningAgent(client=MockClient())
    result = agent.solve("1+1=?", {"idx": 0})

    json.dumps(result, ensure_ascii=False)


def test_user_agent_enables_thinking_mode_when_client_supports_it():
    client = ThinkingMockClient()
    agent = ReasoningAgent(client=client, thinking_mode=True)
    result = agent.solve("1+1=?", {"idx": 0})

    assert result["final_response"] == "2"
    assert client.thinking_mode_values
    assert all(value is True for value in client.thinking_mode_values)


def test_user_agent_repairs_missing_gaussian_curvature_value():
    class CurvatureClient:
        def chat(self, messages, temperature=0.2, max_tokens=4096, thinking_mode=None):
            return (
                '{"final_answer": {"answer": "This value is constant on the unit sphere.", '
                '"answer_type": "text"}, '
                '"verification": {"verification_result": "pass", '
                '"checks": ["Shape operator S = I, so K = det(S) = 1."], '
                '"confidence": 1.0}}'
            )

    agent = ReasoningAgent(client=CurvatureClient())
    result = agent.solve("计算单位球面的高斯曲率。", {"idx": 1, "subject": "微分几何"})

    assert result["final_response"].startswith("K = 1")
