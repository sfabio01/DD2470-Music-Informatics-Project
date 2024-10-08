import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import utils
import json
import numpy as np
from os.path import join as pjoin

metadata_index = ['genre', 'interest', 'year_created']
interest_bins = [0, 2000, 5000, 10000, float('inf')]
interest_bins_labels = ['0-2000', '2000-5000', '5000-10000', '10000+']
year_bins = [2007, 2013, 2018]
year_bins_labels = ['2008-2012', '2013-2017']

class FmaDataset(Dataset):
    def __init__(self, metadata_folder, root_dir, transform=None):
        self.tracks = utils.load('fma_metadata/tracks.csv')
        
        self.genre_track_dict = json.load(open(pjoin(metadata_folder, 'genre_track_dict.json')))
        self.interest_bin_dict = json.load(open(pjoin(metadata_folder, 'interest_bin_dict.json')))
        self.year_created_bin_dict = json.load(open(pjoin(metadata_folder, 'year_created_bin_dict.json')))
        
        self.small = self.tracks[self.tracks['set', 'subset'] <= 'small']
        self.small = self.preprocess_tracks_csv(self.small)

        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.small)

    def __getitem__(self, idx):
        track = self.small.iloc[idx]
        
        positive_id, negative_id = self.pick_positive_negative_sample(track)
        positive_id = str(positive_id).zfill(6)
        negative_id = str(negative_id).zfill(6)
        
        anchor = np.load(pjoin(self.root_dir, f'{track['track_id']}.npy'))
        positive = np.load(pjoin(self.root_dir, f'{positive_id}.npy'))
        negative = np.load(pjoin(self.root_dir, f'{negative_id}.npy'))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return (anchor, positive, negative)
    

    def pick_positive_negative_sample(self, anchor_track):
        selected_metadata = metadata_index[np.random.randint(0, 3)]
        if selected_metadata == 'genre':
            genre = anchor_track['track', 'genre_top']
            genre_tracks = self.genre_track_dict[genre]
            positive_track = genre_tracks[np.random.randint(0, len(genre_tracks))]
            
            other_genres = [g for g in self.genre_track_dict.keys() if g != genre]
            other_genre = other_genres[np.random.randint(0, len(other_genres))]
            other_genre_tracks = self.genre_track_dict[other_genre]
            negative_track = other_genre_tracks[np.random.randint(0, len(other_genre_tracks))]

        elif selected_metadata == 'interest':
            interest = anchor_track['track', 'interest']
            interest_bin = pd.cut([interest], bins=interest_bins, labels=interest_bins_labels)[0]
            interest_tracks = self.interest_bin_dict[interest_bin]
            positive_track = interest_tracks[np.random.randint(0, len(interest_tracks))]

            other_interest_bins = [bin for bin in self.interest_bin_dict.keys() if bin != interest_bin]
            other_interest_bin = other_interest_bins[np.random.randint(0, len(other_interest_bins))]
            other_interest_tracks = self.interest_bin_dict[other_interest_bin]
            negative_track = other_interest_tracks[np.random.randint(0, len(other_interest_tracks))]

        elif selected_metadata == 'year_created':
            year_created = anchor_track['track', 'year_created']
            year_created_bin = pd.cut([year_created], bins=year_bins, labels=year_bins_labels)[0]
            year_created_tracks = self.year_created_bin_dict[year_created_bin]
            positive_track = year_created_tracks[np.random.randint(0, len(year_created_tracks))]

            other_year_created_bins = [bin for bin in self.year_created_bin_dict.keys() if bin != year_created_bin]
            other_year_created_bin = other_year_created_bins[np.random.randint(0, len(other_year_created_bins))]
            other_year_created_tracks = self.year_created_bin_dict[other_year_created_bin]
            negative_track = other_year_created_tracks[np.random.randint(0, len(other_year_created_tracks))]
            
        return positive_track, negative_track
    
    def preprocess_tracks_csv(self, df):
        df['track', 'year_created'] = df['track', 'date_created'].dt.year
        df = df.reset_index(drop=False)
        return df