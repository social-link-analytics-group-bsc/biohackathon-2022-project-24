import ast
import jsonschema

import logging
from typing import Optional


def setup_logger() -> logging.Logger:
    """Setup the logger configuration for consistency."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logger()


json_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "labels": {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "sample": {"type": "array", "items": {"type": "integer"}},
                        "sentence_where_found": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    # "required": ["total", "sample", "sentence_where_found"]
                },
                "male": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "sample": {"type": "array", "items": {"type": "integer"}},
                        "sentence_where_found": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    # "required": ["total", "sample", "sentence_where_found"]
                },
                "female": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "sample": {"type": "array", "items": {"type": "integer"}},
                        "sentence_where_found": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    # "required": ["total", "sample", "sentence_where_found"]
                },
            },
            # "required": ["sample", "male", "female"]
        },
    },
    # "required": ["answer", "labels"]
}



def format_to_json(answer: str) -> Optional[dict]:
    """Safely evaluate a string as Python literal."""
    try:
        return ast.literal_eval(answer)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Invalid answer format: {e}")
        return None


def validate_json(data: dict, schema) -> Optional[dict]:
    """ " pass the answer and validate against a jsonschema"""
    try:
        jsonschema.validate(instance=data, schema=schema)
        return data
    except jsonschema.exceptions.ValidationError as err:
        return None


def remove_none_values(d: Optional[dict]) -> Optional[dict]:
    """Recursively remove None values from a dictionary."""
    try:
        if not isinstance(d, dict):
            return d
        return {k: remove_none_values(v) for k, v in d.items() if v is not None}
    except ValueError:
        return None


def format_answer(answer: str) -> Optional[dict]:
    answer = format_to_json(answer)
    return validate_json(answer, json_schema)
    answer = remove_none_values(answer)
    return answer
