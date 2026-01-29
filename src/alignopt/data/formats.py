"""
Purpose
-------
Typed record representing a single preference example for DPO-style training.

Key behaviors
-------------
- Stores a prompt and two candidate responses (chosen vs rejected).
- Carries a stable identifier for traceability across shuffles/splits/batches.
- Optionally stores arbitrary metadata for debugging, provenance, or analysis.

Parameters
----------
prompt : str
    The input prompt shown to the model.
chosen : str
    The preferred (higher-quality) response for the prompt.
rejected : str
    The non-preferred (lower-quality) response for the prompt.
preference_id : str
    Stable identifier for this preference example (explicitly provided or generated upstream).
meta_data : dict[str, typing.Any] | None
    Optional free-form metadata associated with the record (e.g., source, difficulty, annotator).

Attributes
----------
prompt : str
    Prompt text for the example.
chosen : str
    Preferred response text.
rejected : str
    Non-preferred response text.
preference_id : str
    Stable identifier for the example.
meta_data : dict[str, typing.Any] | None
    Optional metadata payload.

Notes
-----
- Invariants expected by downstream code:
  - prompt, chosen, rejected are non-empty strings.
  - preference_id is a non-empty string and remains stable across program runs for the same example.
- meta_data is treated as opaque; consumers should not assume a fixed schema unless enforced elsewhere.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Preference:
    prompt: str
    chosen: str
    rejected: str
    preference_id: str
    meta_data: Dict[str, Any] | None

