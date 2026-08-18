import json

import pytest

from rgbd_avatar.io import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json_mapping,
    load_jsonl_objects,
    load_yaml_mapping,
)


def test_atomic_json_and_jsonl_round_trip(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.json"
    records_path = tmp_path / "records.jsonl"

    atomic_write_json(mapping_path, {"value": 3})
    atomic_write_jsonl(records_path, [{"index": 0}, {"index": 1}])

    assert load_json_mapping(mapping_path) == {"value": 3}
    assert load_jsonl_objects(records_path) == [
        {"index": 0},
        {"index": 1},
    ]


def test_mapping_loaders_reject_wrong_root_type(tmp_path) -> None:
    json_path = tmp_path / "array.json"
    yaml_path = tmp_path / "array.yaml"
    json_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    yaml_path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_mapping(json_path)
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml_mapping(yaml_path)
