import json
from pathlib import Path

from jsonschema import ValidationError, validate


def load_schema(schema_path=None):
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent / "result_schema.json"
    with open(schema_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_result(result, schema_path=None):
    try:
        validate(instance=result, schema=load_schema(schema_path))
        return {
            "valid": True,
            "error": None,
        }
    except ValidationError as exc:
        return {
            "valid": False,
            "error": exc.message,
        }
