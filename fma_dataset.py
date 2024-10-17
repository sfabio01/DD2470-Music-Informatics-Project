import json
from os.path import join as pjoin
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import utils

# Define constants at module level
METADATA_INDEX = ['genre', 'interest', 'year_created']
INTEREST_BINS = [0, 2000, 5000, 10000, float('inf')]
INTEREST_BINS_LABELS = ['0-2000', '2000-5000', '5000-10000', '10000+']
YEAR_BINS = [2007, 2013, 2018]
YEAR_BINS_LABELS = ['2008-2012', '2013-2017']


class FmaDataset(Dataset):
    def __init__(self, metadata_folder: str, root_dir: str, split: str, transform: Optional[callable] = None, skip_sanity_check: bool = False):
        assert split in ['train', 'val'], "Split must be one of 'train' or 'val'"

        self.split = split

        # Load data only once during initialization
        self.tracks = utils.load(pjoin(metadata_folder, 'tracks.csv'))
        
        # Load metadata dictionaries using a helper method
        self.metadata_dicts = self._load_metadata_dicts(pjoin(metadata_folder, split))
        self.train_val_splits = self._load_train_val_splits(metadata_folder)

        if not skip_sanity_check: assert self._sanity_check()
        
        # Preprocess tracks once during initialization
        self.small = self._preprocess_tracks()
        
        # Store instance variables
        self.root_dir = root_dir
        self.transform = transform
        
        # Pre-calculate valid indices for each category
        self._initialize_category_indices()

        # Load memmap as a singleton
        self.memmap_path = pjoin(root_dir, 'memmap.dat')

        self.filename_to_index = self._load_json_mapping(pjoin(root_dir, 'name_to_index.json'))

    def _sanity_check(self) -> bool:
        """Check if all metadata dictionaries contains the same number of tracks."""
        len_genre = sum(len(tracks) for tracks in self.metadata_dicts['genre'].values())
        len_interest = sum(len(tracks) for tracks in self.metadata_dicts['interest'].values())
        len_year_created = sum(len(tracks) for tracks in self.metadata_dicts['year_created'].values())

        return len_genre == len_interest == len_year_created
    
    def _load_memmap(self) -> np.memmap:
        """Load the memmap file."""
        return np.memmap(self.memmap_path, dtype=np.float16, mode='r', shape=(7997, 1024, 2048, 3))
    
    def _load_json_mapping(self, json_path: str) -> Dict:
        """Load a JSON file and return a dictionary."""
        return json.load(open(json_path))

    def _load_metadata_dicts(self, metadata_folder: str) -> Dict:
        """Load all metadata dictionaries at once."""
        return {
            'genre': json.load(open(pjoin(metadata_folder, 'genre_track_dict.json'))),
            'interest': json.load(open(pjoin(metadata_folder, 'interest_bin_dict.json'))),
            'year_created': json.load(open(pjoin(metadata_folder, 'year_created_bin_dict.json')))
        }

    def _load_train_val_splits(self, metadata_folder: str) -> Dict:
        """Load the train-val splits."""
        return json.load(open(pjoin(metadata_folder, 'train_val_ids.json')))

    def _preprocess_tracks(self) -> pd.DataFrame:
        """Preprocess the tracks DataFrame."""
        small = self.tracks[self.tracks['set', 'subset'] <= 'small'].copy()
        small = small['track'].reset_index(drop=False)
        small['year_created'] = small['date_created'].dt.year
        small = small.rename(columns={'genre_top': 'genre'})
        small = small[~small['track_id'].isin([133297, 99134, 108925])] # remove corrupted tracks
        return small.reset_index(drop=False)

    def _initialize_category_indices(self):
        """Pre-calculate valid indices for each category to avoid repeated computations."""
        self.category_indices = {
            'genre': {genre: np.array(tracks) for genre, tracks in self.metadata_dicts['genre'].items()},
            'interest': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['interest'].items()},
            'year_created': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['year_created'].items()}
        }

    def __len__(self) -> int:
        return len(self.train_val_splits[self.split])

    def _load_track(self, track_id: str) -> np.ndarray:
        """Load track data from npy file."""
        return np.load(pjoin(self.root_dir, f'{track_id.zfill(6)}.npy'))
        # """Load track data from memmap file."""
        # memmap = self._load_memmap()
        # index = self.filename_to_index[f'{track_id.zfill(6)}']
        # x = memmap[index]
        # del memmap
        # return x
        
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
        positive_track = np.random.choice(positive_tracks)
        
        # Get negative sample
        other_bins = [bin for bin in self.category_indices[category].keys() if bin != current_bin]
        other_bin = np.random.choice(other_bins)
        negative_track = np.random.choice(self.category_indices[category][other_bin])
        
        return str(positive_track), str(negative_track)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        track = self.small.iloc[idx]
        track_id = str(track['track_id'])
        
        # Select random category and get samples
        category = np.random.choice(METADATA_INDEX)
        positive_id, negative_id = self._get_samples(track, category)
        
        # Load and transform samples
        samples = [
            torch.tensor(self._load_track(track_id)),
            torch.tensor(self._load_track(positive_id)),
            torch.tensor(self._load_track(negative_id))
        ]
        
        if self.transform:
            samples = [self.transform(sample) for sample in samples]
            
        return tuple(samples)