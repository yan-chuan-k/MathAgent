from __future__ import annotations

from typing import Any, Dict

from .base import MathTool
from .sympy_tool import SafeSympyTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, MathTool] = {}

    def register(self, tool: MathTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> MathTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_SympyAdapter())
    try:
        from .matrix_tool import MatrixTool

        registry.register(MatrixTool())
    except Exception:
        pass
    return registry


class _SympyAdapter(MathTool):
    name = "safe_sympy"

    def __init__(self) -> None:
        self._tool = SafeSympyTool()

    def validate_input(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")

    def run(self, payload: Dict[str, Any], timeout: float = 2.0):
        self.validate_input(payload)
        return self._tool.run_check(payload, claim_id=str(payload.get("claim_id") or "tool_check"))
