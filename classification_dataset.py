import json
from os.path import join as pjoin
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# Genre to index mapping
GENRE_TO_IDX = {
    'Hip-Hop': 0,
    'Pop': 1,
    'Folk': 2,
    'Experimental': 3,
    'Rock': 4,
    'International': 5,
    'Electronic': 6,
    'Instrumental': 7
}


class MyDataset(Dataset):
    def __init__(self, metadata_folder: str, root_dir: str, split: str, transform: Optional[callable] = None, skip_sanity_check: bool = False):
        assert split in ['train', 'val'], "Split must be one of 'train' or 'val'"

        self.split = split
        self.skip_sanity_check = skip_sanity_check

        # Load data only once during initialization
        self.small = pd.read_csv(pjoin(metadata_folder, split, 'small.csv'))

        if not skip_sanity_check: assert self._sanity_check()
        
        # Store instance variables
        self.root_dir = root_dir
        self.transform = transform
        

    def _sanity_check(self) -> bool:
        """Check if all metadata dictionaries contains the same number of tracks."""
        len_genre = sum(len(tracks) for tracks in self.metadata_dicts['genre'].values())
        len_interest = sum(len(tracks) for tracks in self.metadata_dicts['interest'].values())
        len_year_created = sum(len(tracks) for tracks in self.metadata_dicts['year_created'].values())

        return len_genre == len_interest == len_year_created


    def _load_train_val_splits(self, metadata_folder: str) -> Dict:
        """Load the train-val splits."""
        return json.load(open(pjoin(metadata_folder, 'train_val_ids.json')))

    def _initialize_category_indices(self):
        """Pre-calculate valid indices for each category to avoid repeated computations."""
        self.category_indices = {
            'genre': {genre: np.array(tracks) for genre, tracks in self.metadata_dicts['genre'].items()},
            'interest': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['interest'].items()},
            'year_created': {bin: np.array(tracks) for bin, tracks in self.metadata_dicts['year_created'].items()}
        }

    def __len__(self) -> int:
        return len(self.small)

    def _load_track(self, track_id: str) -> np.ndarray:
        """Load track data from npy file."""
        return np.load(pjoin(self.root_dir, f'{track_id.zfill(6)}.npy'))

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
        
        # Get genre
        genre = GENRE_TO_IDX[track['genre_top']]
   
        # Load and transform samples
        sample = torch.from_numpy(self._load_track(track_id))
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, genre
    

if __name__ == '__main__':
    metadata_folder = 'fma_metadata'
    root_dir = 'fma_processed'
    split = 'train'
    dataset = MyDataset(metadata_folder, root_dir, split)
    for i in range(len(dataset)):
        print(dataset[i])
        