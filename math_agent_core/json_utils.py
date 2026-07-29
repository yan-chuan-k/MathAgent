from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from jsonschema import ValidationError, validate

from .schema import normalize_result


@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str] = None


def extract_json_from_text(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")

    cleaned = _strip_code_fence(text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate = _find_first_json_object(cleaned)
    if candidate is None:
        raise ValueError("no JSON object found in model output")
    return json.loads(_light_json_repair(candidate))


def repair_json_locally(
    result: Dict[str, Any],
    problem_id: str,
    model: str,
    backend: str,
    attempts: int = 1,
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    return normalize_result(
        result,
        problem_id=problem_id,
        model=model,
        backend=backend,
        attempts=attempts,
        elapsed_seconds=elapsed_seconds,
    )


def repair_json_with_model(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("model-based JSON repair requires a configured repair agent")


def validate_result(result: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    try:
        validate(instance=result, schema=schema)
        return ValidationResult(valid=True)
    except ValidationError as exc:
        return ValidationResult(valid=False, error=exc.message)


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()


def _find_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _light_json_repair(text: str) -> str:
    repaired = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired
