from math_agent_core.answer_utils import extract_final_answer, normalize_final_response


def test_extract_final_answer_from_json_text():
    text = '{"final_response": "x = 1/2"}'

    assert extract_final_answer(text) == "x = 1/2"


def test_extract_final_answer_strips_prefix():
    assert extract_final_answer("最终答案：72") == "72"


def test_normalize_empty_answer_falls_back():
    assert normalize_final_response("") == "无法确定"
