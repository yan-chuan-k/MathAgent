from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response


def test_extract_final_answer_from_json_text():
    text = '{"final_response": "x = 1/2"}'

    assert extract_final_answer(text) == "x = 1/2"


def test_extract_final_answer_strips_chinese_prefix():
    assert extract_final_answer("最终答案：72") == "72"
    assert extract_final_answer("答案：x = 1/2") == "x = 1/2"


def test_normalize_empty_answer_falls_back():
    assert normalize_final_response("") == DEFAULT_FALLBACK


def test_extract_final_answer_from_boxed_latex():
    assert extract_final_answer(r"因此 \boxed{42}") == "42"


def test_extract_final_answer_from_nested_boxed_latex():
    assert extract_final_answer(r"最终得到 \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_final_answer_from_nested_final_answer_json():
    text = '{"final_answer": {"answer": "e^{-t}"}}'

    assert extract_final_answer(text) == "e^{-t}"


def test_extract_final_answer_from_last_marked_segment():
    text = "先计算若干步骤。由此可知最终答案为：最大值 2，最小值 -2。"

    assert extract_final_answer(text) == "最大值 2，最小值 -2。"
