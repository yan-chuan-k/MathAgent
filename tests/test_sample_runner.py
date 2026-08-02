import json
import subprocess
import sys
from pathlib import Path

from main import is_successful_output


def test_sample_runner_outputs_idx_files(tmp_path):
    input_file = tmp_path / "dev.jsonl"
    output_dir = tmp_path / "outputs"
    input_file.write_text('{"idx": 0, "problem": "1+1?"}\n', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input_file",
            str(input_file),
            "--output_dir",
            str(output_dir),
            "--mock",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    result_path = output_dir / "0.json"
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["final_response"]


def test_runner_skips_only_successful_nonempty_outputs(tmp_path):
    output_path = tmp_path / "0.json"

    output_path.write_text('{"status": "error", "final_response": ""}', encoding="utf-8")
    assert is_successful_output(output_path) is False

    output_path.write_text('{"status": "success", "final_response": "2"}', encoding="utf-8")
    assert is_successful_output(output_path) is True
