import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import utils
import json
import numpy as np
from os.path import join as pjoin
from typing import Optional, Tuple, Dict, List
from functools import lru_cache

# Define constants at module level
METADATA_INDEX = ['genre', 'interest', 'year_created']
INTEREST_BINS = [0, 2000, 5000, 10000, float('inf')]
INTEREST_BINS_LABELS = ['0-2000', '2000-5000', '5000-10000', '10000+']
YEAR_BINS = [2007, 2013, 2018]
YEAR_BINS_LABELS = ['2008-2012', '2013-2017']

class FmaDataset(Dataset):
    def __init__(self, metadata_folder: str, root_dir: str, transform: Optional[callable] = None):
        # Load data only once during initialization
        self.tracks = utils.load(pjoin(metadata_folder, 'tracks.csv'))
        
        # Load metadata dictionaries using a helper method
        self.metadata_dicts = self._load_metadata_dicts(metadata_folder)
        
        # Preprocess tracks once during initialization
        self.small = self._preprocess_tracks()
        
        # Store instance variables
        self.root_dir = root_dir
        self.transform = transform
        
        # Pre-calculate valid indices for each category
        self._initialize_category_indices()

    def _load_metadata_dicts(self, metadata_folder: str) -> Dict:
        """Load all metadata dictionaries at once."""
        return {
            'genre': json.load(open(pjoin(metadata_folder, 'genre_track_dict.json'))),
            'interest': json.load(open(pjoin(metadata_folder, 'interest_bin_dict.json'))),
            'year_created': json.load(open(pjoin(metadata_folder, 'year_created_bin_dict.json')))
        }

    def _preprocess_tracks(self) -> pd.DataFrame:
        """Preprocess the tracks DataFrame."""
        small = self.tracks[self.tracks['set', 'subset'] <= 'small'].copy()
        small = small['track']
        small['year_created'] = small['date_created'].dt.year
        small = small.rename(columns={'genre_top': 'genre'})
        return small.reset_index(drop=False)

    def _initialize_category_indices(self):
        """Pre-calculate valid indices for each category to avoid repeated computations."""
        self.category_indices = {
            'genre': {genre: np.array(tracks) for genre, tracks in self.metadata_dicts['genre'].items()},
            'interest': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['interest'].items()},
            'year_created': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['year_created'].items()}
        }

    def __len__(self) -> int:
        return len(self.small)

    @lru_cache()
    def _load_track(self, track_id: str) -> np.ndarray:
        """Load and cache track data."""
        return np.load(pjoin(self.root_dir, f'{track_id.zfill(6)}.npy'))
    
    def _load_tracks(self, track_ids: str) -> np.ndarray:
        """Load multiple tracks and concatenate them."""
        return np.concatenate([self._load_track(track_id) for track_id in track_ids])

    def _get_bin_for_value(self, value: float, category: str) -> str:
        """Get the appropriate bin for a value in a category."""
        if category == 'interest':
            return pd.cut([value], bins=INTEREST_BINS, labels=INTEREST_BINS_LABELS)[0]
        elif category == 'year_created':
            return pd.cut([value], bins=YEAR_BINS, labels=YEAR_BINS_LABELS)[0]
        return value  # For genre, return as is

    def _get_samples(self, anchor_track: pd.Series, category: str) -> Tuple[str, str]:
        """Get positive and negative samples for a given category."""
        value = anchor_track[category]
        current_bin = self._get_bin_for_value(value, category)
        
        # Get positive sample
        positive_tracks = self.category_indices[category][current_bin]

        assert len(positive_tracks) >= 20, f'Not enough positive samples for {category} {current_bin}'

        selected_positive_tracks = np.random.choice(positive_tracks, size=20)
        
        # Get negative sample
        other_bins = [bin for bin in self.category_indices[category].keys() if bin != current_bin]
        other_bin = np.random.choice(other_bins)

        assert len(self.category_indices[category][other_bin]) >= 20, f'Not enough negative samples for {category} {other_bin}'

        selected_negative_tracks = np.random.choice(self.category_indices[category][other_bin], size=20)
        
        return map(str, selected_positive_tracks), map(str, selected_negative_tracks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        track = self.small.iloc[idx]
        track_id = str(track['track_id'])
        
        # Select random category and get samples
        category = np.random.choice(METADATA_INDEX)
        positive_ids, negative_ids = self._get_samples(track, category)
        
        # Load and transform samples
        samples = [
            self._load_track(track_id),
            self._load_tracks(positive_ids),
            self._load_tracks(negative_ids)
        ]
        
        if self.transform:
            samples[0] = self.transform(samples[0])
            for i in range(len(samples[1])):
                samples[1][i] = self.transform(samples[1][i])
                samples[2][i] = self.transform(samples[2][i])
    
            
        return tuple(samples)