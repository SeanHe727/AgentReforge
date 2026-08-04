from __future__ import annotations

from pydantic import BaseModel

from agentreforge.llm.parse import parse_json_model


class Payload(BaseModel):
    value: int


def test_parser_ignores_text_after_first_complete_json_object():
    parsed = parse_json_model('{"value": 7} trailing characters {"value": 9}', Payload)

    assert parsed.value == 7


def test_parser_ignores_trailing_text_inside_json_fence():
    parsed = parse_json_model(
        '```json\n{"value": 11} explanatory suffix\n```',
        Payload,
    )

    assert parsed.value == 11


def test_parser_still_rejects_invalid_first_object():
    try:
        parse_json_model('{"value": } {"value": 9}', Payload)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("invalid first object must not be skipped")
