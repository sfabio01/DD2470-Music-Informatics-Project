from multiprocessing import Pool
import librosa
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom
from tqdm import tqdm
import gc
import json

CORRUPTED_FILES = []
song_to_feature={}

def load_audio_mono(file_path:str)->tuple[np.ndarray,int]:
    """Load an audio file and ensure it's mono using librosa."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def process_audio_file(file_path:str)->np.ndarray:
    """Process an audio file to compute spectrogram, chromagram, and tempogram."""
    try:
        # Load audio
        y, sr = load_audio_mono(file_path)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        song_to_feature[file_path.split('/')[-1]] = [zcr,spec_centroid]
        
    except Exception as e:
        CORRUPTED_FILES.append(file_path)
        print(f"Error processing {file_path}: {e}")

def process_and_store(args):
    i, file_path, memmap_path, memmap_shape = args
    process_audio_file(file_path)

    

def process_files_in_parallel(file_list: list[str], output_dir: str, num_workers: int = 4) -> None:
    """Process multiple audio files in parallel using multi-processing with a progress bar."""
    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(output_dir, "memmap.dat")
    memmap_shape = (len(file_list), 1024, 2048, 3)
    
    with Pool(processes=num_workers) as pool:
        args_list = [(i, file_path, memmap_path, memmap_shape) for i, file_path in enumerate(file_list)]
        for _ in tqdm(pool.imap_unordered(process_and_store, args_list), total=len(file_list), desc="Processing audio files"):
            pass

    print(song_to_feature)
    json.dump(song_to_feature, open(os.path.join(output_dir, "song_to_feature.json"), "w"))


def librosa_load_wrapper(file_path):
    try:
        y, sr = librosa.load(file_path)
        return file_path, y is not None and len(y) > 0
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return file_path, False
    
def sanity_check(file_list:list[str], num_workers:int = 4)->tuple[list[str], list[str]]:
    """Filter the file_list to only include valid files and output a new file_list."""
    valid_files = []
    invalid_files = []

    with Pool(processes=num_workers) as pool:
        for file_path, is_valid in tqdm(pool.imap_unordered(librosa_load_wrapper, file_list), total=len(file_list)):
            if is_valid:
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)

    print(f"Found {len(valid_files)} valid files out of {len(file_list)} total files.")
    return valid_files, invalid_files
    
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Process audio files to compute spectrogram, chromagram, and tempogram.")
    parser.add_argument("input_dir", help="Directory containing input audio files, should contain subdirectories with audio files")
    parser.add_argument("output_dir", help="Directory to save the processed data")
    parser.add_argument("--file_format", type=str, default="mp3", help="File format of audio files, e.g., 'mp3' 'wav'")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    start_time = time.time()
    print("Starting audio processing...")

    file_list = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(f".{args.file_format}"):
                file_list.append(os.path.join(root, file))
                
    print(f"Found {len(file_list)} files to process.")

    valid_files, invalid_files = sanity_check(file_list, num_workers=args.num_workers)
    process_files_in_parallel(valid_files, args.output_dir, num_workers=args.num_workers)

    end_time = time.time()
    print("Audio processing completed.")
    print(f"Total time taken: {end_time - start_time} seconds")
    
    print()
    print(invalid_files)
    print(CORRUPTED_FILES)

