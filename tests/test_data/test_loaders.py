"""
Purpose
-------
Unit tests for the alignopt.data.loaders module.

These tests validate that the data-loading utilities correctly:
1) read JSONL preference records into strongly-typed Preference objects,
2) skip invalid JSON lines and rows missing required fields,
3) generate deterministic, content-based preference IDs when missing,
4) perform deterministic train/eval splits given a seed,
5) batch Preference objects into (prompt, chosen)/(prompt, rejected) pairs while retaining IDs.

Key behaviors
-------------
- Reads a JSONL file where each line is a JSON object.
- Required keys per record: "prompt", "chosen", "rejected".
- Optional keys per record: "id", "meta_data".
- Invalid JSON lines are skipped.
- Records missing any required field are skipped.
- load_and_split_data uses NumPy Generator shuffle; determinism depends on seed.

Conventions
-----------
- Preference IDs are treated as stable identifiers for debugging and traceability.
- If "id" is missing, the module generates an ID from (prompt, chosen, rejected) only.
  This implies duplicates with identical triples will share the same ID.
- The split is a simple shuffled partition, not stratified.
- "training_fraction" must be in [0, 1].

Downstream usage
----------------
Run with:
    pytest -q

Adjust imports if your package/module paths differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from alignopt.data.formats import Preference
from alignopt.data.loaders import DataBatch, batch_and_collect_data, load_and_split_data


def _write_jsonl(path: Path, records: Sequence[str]) -> None:
    """
    Write JSONL content to disk.

    Parameters
    ----------
    path : pathlib.Path
        Destination path to write the JSONL file.
    records : Sequence[str]
        Each element is written as one line to the file.

    Returns
    -------
    None
        Writes to disk.

    Raises
    ------
    OSError
        If the file cannot be written.

    Notes
    -----
    - The caller controls whether each line is valid JSON or not.
    """
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _make_record(
    prompt: str | None,
    chosen: str | None,
    rejected: str | None,
    record_id: str | None = None,
    meta_data: Dict[str, Any] | None = None,
) -> str:
    """
    Create a single JSONL record line.

    Parameters
    ----------
    prompt : str | None
        Prompt field; if None, the key is still written with value None.
    chosen : str | None
        Chosen completion field; if None, the key is still written with value None.
    rejected : str | None
        Rejected completion field; if None, the key is still written with value None.
    record_id : str | None, optional
        Optional "id" field; if None, it is omitted.
    meta_data : Dict[str, Any] | None, optional
        Optional "meta_data" field; if None, it is omitted.

    Returns
    -------
    str
        A JSON string representing one record.

    Notes
    -----
    - This helper deliberately allows None values to exercise skipping logic.
    """
    d: Dict[str, Any] = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if record_id is not None:
        d["id"] = record_id
    if meta_data is not None:
        d["meta_data"] = meta_data
    return json.dumps(d, sort_keys=True)


def test_load_and_split_data_rejects_invalid_training_fraction(tmp_path: Path) -> None:
    """
    Validate that training_fraction is range-checked.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory fixture.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If ValueError is not raised for invalid fractions.

    Notes
    -----
    - This ensures the function fails fast on invalid user inputs.
    """
    p: Path = tmp_path / "data.jsonl"
    _write_jsonl(p, [_make_record("p", "c", "r")])

    with pytest.raises(ValueError):
        _ = load_and_split_data(str(p), training_fraction=-0.1, random_seed=0)

    with pytest.raises(ValueError):
        _ = load_and_split_data(str(p), training_fraction=1.1, random_seed=0)


def test_read_skips_invalid_json_and_missing_required_fields(tmp_path: Path) -> None:
    """
    Validate that invalid JSON lines and incomplete records are skipped.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the loader includes invalid/incomplete rows.

    Notes
    -----
    - The file contains:
      - one invalid JSON line,
      - three incomplete lines (missing a required field),
      - two valid lines.
    - Only the two valid lines should remain.
    """
    p: Path = tmp_path / "data.jsonl"
    lines: List[str] = [
        "this is not json",
        _make_record(None, "c1", "r1"),
        _make_record("p2", None, "r2"),
        _make_record("p3", "c3", None),
        _make_record("p_ok_1", "c_ok_1", "r_ok_1", meta_data={"source": "x"}),
        _make_record("p_ok_2", "c_ok_2", "r_ok_2", record_id="explicit-id-2"),
    ]
    _write_jsonl(p, lines)

    train, eval_ = load_and_split_data(str(p), training_fraction=1.0, random_seed=123)
    assert len(train) == 2
    assert len(eval_) == 0

    prompts: List[str] = [x.prompt for x in train]
    assert "p_ok_1" in prompts
    assert "p_ok_2" in prompts


def test_generated_ids_are_deterministic_for_same_content(tmp_path: Path) -> None:
    """
    Validate that missing IDs are generated deterministically from content.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the same content yields different generated IDs.

    Notes
    -----
    - Two identical records without an explicit "id" should produce the same generated ID.
    - This is expected given content-based ID generation.
    """
    p: Path = tmp_path / "data.jsonl"
    rec: str = _make_record("same prompt", "same chosen", "same rejected")
    _write_jsonl(p, [rec, rec])

    train, _ = load_and_split_data(str(p), training_fraction=1.0, random_seed=0)
    assert len(train) == 2

    id0: str | None = train[0].preference_id
    id1: str | None = train[1].preference_id
    assert id0 is not None
    assert id1 is not None
    assert id0 == id1


def test_load_and_split_data_is_deterministic_given_seed(tmp_path: Path) -> None:
    """
    Validate that shuffling + split are deterministic for a fixed seed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If repeated calls with the same seed yield different partitions.

    Notes
    -----
    - We verify determinism by comparing the ordered list of IDs in train/eval across runs.
    - This assumes the implementation uses np.random.default_rng(seed) + rng.shuffle(list).
    """
    p: Path = tmp_path / "data.jsonl"
    lines: List[str] = [
        _make_record("p0", "c0", "r0", record_id="id0"),
        _make_record("p1", "c1", "r1", record_id="id1"),
        _make_record("p2", "c2", "r2", record_id="id2"),
        _make_record("p3", "c3", "r3", record_id="id3"),
        _make_record("p4", "c4", "r4", record_id="id4"),
    ]
    _write_jsonl(p, lines)

    train_a, eval_a = load_and_split_data(str(p), training_fraction=0.6, random_seed=42)
    train_b, eval_b = load_and_split_data(str(p), training_fraction=0.6, random_seed=42)

    ids_train_a: List[str] = [x.preference_id for x in train_a if x.preference_id is not None]
    ids_eval_a: List[str] = [x.preference_id for x in eval_a if x.preference_id is not None]
    ids_train_b: List[str] = [x.preference_id for x in train_b if x.preference_id is not None]
    ids_eval_b: List[str] = [x.preference_id for x in eval_b if x.preference_id is not None]

    assert ids_train_a == ids_train_b
    assert ids_eval_a == ids_eval_b
    assert len(train_a) == 3
    assert len(eval_a) == 2


def test_batch_and_collect_data_builds_pairs_and_ids() -> None:
    """
    Validate that batching produces aligned chosen/rejected pairs and retains IDs.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If prompts are not aligned or IDs are missing/misordered.

    Notes
    -----
    - The chosen and rejected batches must use the same prompts in the same order.
    """
    raw: List[Preference] = [
        Preference(prompt="p0", chosen="c0", rejected="r0", preference_id="id0", meta_data=None),
        Preference(prompt="p1", chosen="c1", rejected="r1", preference_id="id1", meta_data={"k": 1}),
    ]

    batch: DataBatch = batch_and_collect_data(raw)

    assert isinstance(batch, DataBatch)
    assert batch.chosen == [("p0", "c0"), ("p1", "c1")]
    assert batch.rejected == [("p0", "r0"), ("p1", "r1")]
    assert batch.preference_ids == ["id0", "id1"]


def test_batch_and_collect_data_empty_returns_empty_databatch() -> None:
    """
    Validate that batching an empty list returns an empty DataBatch.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the function returns the wrong type or non-empty fields.

    Notes
    -----
    None
    """
    batch: DataBatch = batch_and_collect_data([])

    assert isinstance(batch, DataBatch)
    assert batch.chosen == []
    assert batch.rejected == []
    assert batch.preference_ids == []
