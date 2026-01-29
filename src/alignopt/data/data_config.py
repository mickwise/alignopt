"""
Purpose
-------
Centralized constants for the data layer.

Key behaviors
-------------
- Defines a fixed UUID namespace used for deterministic, content-based preference IDs.
- Defines the default RNG seed used for reproducible data shuffling/splitting.
- Defines the text encoding used when reading JSONL datasets from disk.

Conventions
-----------
- NAME_SPACE is constant across runs and across machines; changing it will change all generated UUIDv5 IDs.
- RANDOM_SEED is the directory-wide default seed; callers may override per function for experiments.
- TEXT_ENCODING is used whenever reading/writing dataset files to ensure consistent hashing and parsing.

Downstream usage
----------------
- loaders._generate_preference_id(...) uses NAME_SPACE (UUIDv5) to produce stable IDs.
- loaders.load_and_split_data(...) uses RANDOM_SEED as the default seed for deterministic splits.
- Any file IO in the data layer should use TEXT_ENCODING to avoid platform-dependent behavior.
"""

import uuid

NAME_SPACE: uuid.UUID = uuid.UUID("0df0b2a5-e0ba-426a-ab31-4ea724f356ce")
RANDOM_SEED: int = 42
TEXT_ENCODING: str = "utf-8"
