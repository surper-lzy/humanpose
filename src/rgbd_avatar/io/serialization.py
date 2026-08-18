"""Validated JSON/YAML reads and recoverable atomic writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml


def resolve_path(value: str | Path, *, relative_to: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(relative_to).expanduser().resolve() / path
    return path.resolve()


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved}.")
    return payload


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"YAML file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}.")
    return payload


def load_jsonl_objects(
    path: str | Path,
    *,
    require_nonempty: bool = True,
) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {resolved}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object at {resolved}:{line_number}."
                )
            records.append(payload)
    if require_nonempty and not records:
        raise ValueError(f"No records found in {resolved}.")
    return records


def _atomic_text_write(path: Path, writer: Any) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{resolved.stem}.",
            suffix=resolved.suffix or ".tmp",
            dir=resolved.parent,
            encoding="utf-8",
            delete=False,
        ) as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        os.replace(temporary, resolved)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    def write(file: Any) -> None:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")

    _atomic_text_write(Path(path), write)


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    def write(file: Any) -> None:
        for record in records:
            json.dump(record, file, ensure_ascii=False, allow_nan=False)
            file.write("\n")

    _atomic_text_write(Path(path), write)
