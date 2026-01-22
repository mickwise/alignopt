from dataclasses import dataclass

@dataclass
class Preference:
    prompt: str
    chosen: str
    rejected: str
