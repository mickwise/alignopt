"""
Purpose
-------
Load, validate, and batch preference-format data for DPO-style training and evaluation.

Key behaviors
-------------
- Reads JSONL preference records from disk into a typed in-memory representation.
- Skips invalid JSON lines and records missing required fields (prompt/chosen/rejected).
- Generates deterministic, content-based preference IDs when no explicit ID is provided.
- Splits data into train/eval partitions using a seeded RNG for reproducibility.
- Converts lists of Preference objects into (prompt, response) pair batches for chosen/rejected.

Conventions
-----------
- Input file format is JSONL (one JSON object per line).
- Required fields are: "prompt", "chosen", "rejected".
- Optional fields are: "id" (string), "meta_data" (JSON object or null).
- If "id" is missing, IDs are derived deterministically from (prompt, chosen, rejected)
  via a stable hashing + UUID5 scheme under a fixed project namespace.
- Shuffling + split are deterministic for a fixed random_seed, but the exact split depends
  on input file ordering and the parsed subset of valid records.

Downstream usage
----------------
- Use load_and_split_data(...) to obtain train/eval Preference lists.
- Use batch_and_collect_data(...) to convert a list of Preference into a DataBatch
  suitable for logprob/NLL backends that expect (prompt, completion) string pairs.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
import pathlib
from hashlib import sha256
import uuid
import numpy as np
from alignopt.data.data_config import RANDOM_SEED, NAME_SPACE, TEXT_ENCODING
from alignopt.data.formats import Preference

@dataclass
class DataBatch:
    """
    Purpose
    -------
    Container for a batched set of preference examples in pair format.

    Key behaviors
    -------------
    - Stores two aligned lists of (prompt, response) string pairs:
      one for the chosen responses and one for the rejected responses.
    - Stores the per-example preference_ids aligned to chosen/rejected.

    Parameters
    ----------
    chosen : list[tuple[str, str]]
        List of (prompt, chosen_response) pairs.
    rejected : list[tuple[str, str]]
        List of (prompt, rejected_response) pairs.
    preference_ids : list[str]
        List of stable identifiers aligned with the corresponding chosen/rejected entries.

    Attributes
    ----------
    chosen : list[tuple[str, str]]
        Batched chosen pairs.
    rejected : list[tuple[str, str]]
        Batched rejected pairs.
    preference_ids : list[str]
        Stable IDs aligned to batch rows.

    Notes
    -----
    - Alignment invariant: len(chosen) == len(rejected) == len(preference_ids).
    - IDs are included to support debugging, traceability, and test assertions.
    """

    chosen: List[Tuple[str, str]]
    rejected: List[Tuple[str, str]]
    preference_ids: List[str]


def load_and_split_data(
        data_path_str: str,
        training_fraction: float = 0.8,
        random_seed: int = RANDOM_SEED
    ) -> Tuple[List[Preference], List[Preference]]:
    """
    Load a JSONL preference dataset, shuffle deterministically, and split into train/eval partitions.

    Parameters
    ----------
    data_path_str : str
        Path to a JSONL file containing preference records (one JSON object per line).
    training_fraction : float, optional
        Fraction of the valid parsed dataset assigned to the training split.
        Must satisfy 0.0 <= training_fraction <= 1.0. Default is 0.8.
    random_seed : int, optional
        Seed used to initialize the RNG that shuffles the dataset before splitting.
        Default is alignopt.data.data_config.RANDOM_SEED.

    Returns
    -------
    tuple[list[Preference], list[Preference]]
        (train_preferences, eval_preferences) where both are lists of Preference objects.

    Raises
    ------
    ValueError
        If training_fraction is outside [0.0, 1.0].

    Notes
    -----
    - Determinism: for a fixed (file contents, file line ordering, training_fraction, random_seed),
      the output partitions and their internal ordering are deterministic.
    - Only valid records (well-formed JSON and containing prompt/chosen/rejected) are considered.
    - The split boundary uses integer truncation: int(training_fraction * N).
    """

    if training_fraction > 1.0 or training_fraction < 0.0:
        raise ValueError("Training fraction must be between 0 and 1.")
    rng: np.random.Generator = np.random.default_rng(seed=random_seed)
    preference_list: List[Preference] = _read_data(data_path_str)
    sample_size: int = len(preference_list)
    training_amount: int = int(training_fraction*sample_size)
    rng.shuffle(preference_list)
    return preference_list[:training_amount], preference_list[training_amount:]


def batch_and_collect_data(raw_data: List[Preference]) -> DataBatch:
    """
    Convert a list of Preference objects into a DataBatch of chosen/rejected (prompt, response) pairs.

    Parameters
    ----------
    raw_data : list[Preference]
        The preference examples to batch.

    Returns
    -------
    DataBatch
        A DataBatch where:
        - chosen contains (prompt, chosen_response) pairs,
        - rejected contains (prompt, rejected_response) pairs,
        - preference_ids contains the aligned preference IDs.

    Raises
    ------
    None

    Notes
    -----
    - For empty input, returns DataBatch([], [], []) to preserve type stability.
    - The returned lists preserve the input ordering of raw_data.
    """

    if len(raw_data) == 0:
        return DataBatch([], [], [])
    chosen: List[Tuple[str, str]] = []
    rejected: List[Tuple[str, str]] = []
    preference_ids: List[str] = []
    for preference in raw_data:
        prompt: str = preference.prompt
        chosen.append((prompt, preference.chosen))
        rejected.append((prompt, preference.rejected))
        preference_ids.append(preference.preference_id)
    return DataBatch(
        chosen,
        rejected,
        preference_ids
    )


def _read_data(data_path_str: str) -> List[Preference]:
    """
    Read a JSONL preference file from disk and return a list of valid Preference objects.

    Parameters
    ----------
    data_path_str : str
        Path to the JSONL file to read.

    Returns
    -------
    list[Preference]
        List of parsed Preference objects for records that pass validation.

    Raises
    ------
    None

    Notes
    -----
    - Records are skipped if:
        - the line is not valid JSON, or
        - any required field ("prompt", "chosen", "rejected") is missing or falsy.
    - If a record lacks an explicit "id", a deterministic content-based ID is generated.
    - This function is intentionally forgiving to support noisy datasets; callers that need strict
      validation should add a stricter loader variant.
    """

    data_path: pathlib.Path = pathlib.Path(data_path_str)
    preference_list: List[Preference] = []
    with data_path.open(encoding=TEXT_ENCODING) as f:
        for raw_record in f:
            try:
                record = json.loads(raw_record)
            except Exception:
                continue
            prompt: str = record.get("prompt")
            chosen: str = record.get("chosen")
            rejected: str = record.get("rejected")
            if prompt and chosen and rejected:
                preference_id: str | None = record.get("id")
                preference_id = (
                    preference_id if preference_id
                    else _generate_preference_id(prompt, chosen, rejected)
                )
                preference_list.append(
                    Preference(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        preference_id=preference_id,
                        meta_data= record.get("meta_data")
                    )
                )
    return preference_list


def _generate_preference_id(prompt: str, chosen: str, rejected: str) -> str:
    """
    Generate a deterministic preference ID from the content of a preference record.

    Parameters
    ----------
    prompt : str
        The prompt text.
    chosen : str
        The preferred response text.
    rejected : str
        The non-preferred response text.

    Returns
    -------
    str
        A stable UUID5 string derived from the record content under the project namespace.

    Raises
    ------
    None

    Notes
    -----
    - This ID is content-addressed: identical (prompt, chosen, rejected) triples yield the same ID.
    - Collisions are extremely unlikely due to hashing + UUID5, but duplicates are allowed by design
      (e.g., repeated identical examples in a dataset).
    - ID stability depends on:
        - stable JSON serialization (sort_keys=True),
        - the configured TEXT_ENCODING,
        - the configured NAME_SPACE UUID.
    """

    id_json: Dict[str, str] = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    id_sha: str = sha256(json.dumps(id_json, sort_keys=True).encode(TEXT_ENCODING)).hexdigest()
    return str(uuid.uuid5(NAME_SPACE, id_sha))
