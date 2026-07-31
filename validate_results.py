import json
import sys
from jsonschema import validate, ValidationError


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_jsonl(results_path, schema_path):
    schema = load_schema(schema_path)

    total = 0
    passed = 0
    failed = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                data = json.loads(line)
                validate(instance=data, schema=schema)
                passed += 1
                print(f"第 {line_number} 行通过：{data.get('problem_id')}")

            except json.JSONDecodeError as e:
                failed += 1
                print(f"第 {line_number} 行 JSON 解析失败：{e}")

            except ValidationError as e:
                failed += 1
                problem_id = data.get("problem_id", "未知题号")
                print(f"第 {line_number} 行格式校验失败：{problem_id}")
                print(f"原因：{e.message}")

    print("\n校验完成")
    print(f"总数：{total}")
    print(f"通过：{passed}")
    print(f"失败：{failed}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    schema_path = sys.argv[2] if len(sys.argv) > 2 else "result_schema.json"
    validate_jsonl(
        results_path=results_path,
        schema_path=schema_path
    )
