from __future__ import annotations

from typing import Any, Tuple


class FormatError(Exception):
    pass


class BaseFormatter:
    """Minimal formatter base compatible with AsyncLLM.call_with_format.

    Subclasses should implement schema-specific prompt preparation and
    response validation/parse.
    """

    def prepare_prompt(self, prompt: str) -> str:
        return prompt

    def validate_response(self, response_text: str) -> Tuple[bool, Any]:
        return True, response_text

    def format_error_message(self) -> str:
        return "Response does not match expected format"


class JSONListOfActionSpecsFormatter(BaseFormatter):
    """Validate that response is a JSON list of action specs.

    Each item should include: name, description, parameters (list or object).
    Returns (True, list[dict]) on success.
    """

    def __init__(self, required_fields: tuple[str, ...] = ("name", "description", "parameters")):
        self.required_fields = required_fields
        self._last_error = ""

    def prepare_prompt(self, prompt: str) -> str:
        guide = (
            "You must output ONLY a valid JSON array of action specs. "
            "Each item must contain keys: name, description, parameters. "
            "Do not include any extra commentary.\n"
        )
        return guide + prompt

    def validate_response(self, response_text: str):
        import json
        try:
            data = json.loads(response_text)
        except Exception as e:
            self._last_error = f"Invalid JSON: {e}"
            return False, None
        if not isinstance(data, list):
            self._last_error = "Top-level JSON is not a list"
            return False, None
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self._last_error = f"Item {i} is not an object"
                return False, None
            for f in self.required_fields:
                if f not in item:
                    self._last_error = f"Item {i} missing field: {f}"
                    return False, None
        return True, data

    def format_error_message(self) -> str:
        return self._last_error or super().format_error_message()
