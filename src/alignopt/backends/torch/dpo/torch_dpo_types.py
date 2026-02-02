from typing import List
import torch
from alignopt.data.formats import Preference

class PreferenceDataset(torch.utils.data.Dataset):

    def __init__(self, preferences: List[Preference]) -> None:
        self.preferences = preferences

    def __len__(self) -> int:
        return len(self.preferences)
    
    def __getitem__(self, index) -> Preference:
        return self.preferences[index]