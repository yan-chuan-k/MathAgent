import json
from pathlib import Path
from jsonschema import validate, ValidationError


class JsonResultValidator:
    def __init__(self, schema_path="result_schema.json"):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()

    def _load_schema(self):
        if not self.schema_path.exists():
            raise FileNotFoundError(f"未找到 Schema 文件：{self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_result(self, result):
        """
        返回：
        {
            "valid": True/False,
            "error": ""
        }
        """
        try:
            validate(instance=result, schema=self.schema)
            return {
                "valid": True,
                "error": ""
            }
        except ValidationError as e:
            return {
                "valid": False,
                "error": e.message
            }
