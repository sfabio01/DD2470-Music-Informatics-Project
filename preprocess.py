from multiprocessing import Pool
import librosa
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom
from tqdm import tqdm

CORRUPTED_FILES = []

def load_audio_mono(file_path:str)->tuple[np.ndarray,int]:
    """Load an audio file and ensure it's mono using librosa."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def process_audio_file(file_path:str)->np.ndarray:
    """Process an audio file to compute spectrogram, chromagram, and tempogram."""
    try:
        # Load audio
        y, sr = load_audio_mono(file_path)

        # Compute STFT and take magnitude
        S_complex = librosa.stft(y)
        S_magnitude = np.abs(S_complex)
        
        # Compute Spectrogram (Log-amplitude)
        log_S = librosa.amplitude_to_db(S_magnitude)

        # Compute Chromagram using the magnitude spectrogram
        chroma = librosa.feature.chroma_stft(S=S_magnitude, sr=sr)

        # Compute Tempogram from onset envelope
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)

        # Align time axes
        min_frames = min(log_S.shape[1], chroma.shape[1], tempogram.shape[1])
        log_S = log_S[:, :min_frames]
        chroma = chroma[:, :min_frames]
        tempogram = tempogram[:, :min_frames]

        # Resize chroma and tempogram to match spectrogram's frequency bins
        n_freq_bins = log_S.shape[0]

        # Use zoom to resize chroma and tempogram
        chroma_resized = zoom(chroma, (n_freq_bins / chroma.shape[0], 1), order=1)
        tempogram_resized = zoom(tempogram, (n_freq_bins / tempogram.shape[0], 1), order=1)

        # Ensure that resizing didn't introduce complex values
        assert np.isrealobj(chroma_resized), "Chroma resized contains complex values!"
        assert np.isrealobj(tempogram_resized), "Tempogram resized contains complex values!"

        # Stack features into a NumPy array (shape: [n_freq_bins, min_frames, 3])
        data = np.stack([log_S, chroma_resized, tempogram_resized], axis=-1)
        
        # Cut to 2048 x 1024
        data = data[:1024, :2048, :]
        
        # Pad with zeros to ensure it's 1024 x 2048
        data = np.pad(data, ((0, 1024 - data.shape[0]), (0, 2048 - data.shape[1]), (0, 0)))
        
        # rotate 180 degrees
        data = np.rot90(data, 2)  

        return data
        
    except Exception as e:
        CORRUPTED_FILES.append(file_path)
        print(f"Error processing {file_path}: {e}")

def process_and_store(i_file):
    i, file_path = i_file
    data = process_audio_file(file_path)
    if data is not None:
        memmap[i] = data.astype(np.float16)
        del data

def process_files_in_parallel(file_list: list[str], output_dir: str, num_workers: int = 4) -> None:
    """Process multiple audio files in parallel using multi-processing with a progress bar."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "memmap.dat")
    memmap_shape = (len(file_list), 1024, 2048, 3)
    memmap = np.memmap(path, dtype=np.float16, mode='w+', shape=memmap_shape)

    def init_shared_memmap(shared_memmap):
        global memmap
        memmap = shared_memmap

    with Pool(processes=num_workers, initializer=init_shared_memmap, initargs=(memmap,)) as pool:
        steps = 0
        for _ in tqdm(pool.imap_unordered(process_and_store, enumerate(file_list)), total=len(file_list), desc="Processing audio files"):
            steps += 1
            if steps % 1000 == 0:
                memmap.flush()  # otherwise we are storing it in memory

    memmap.flush()
    del memmap

def librosa_load_wrapper(file_path):
    try:
        y, sr = librosa.load(file_path)
        if len(y) / sr < 29.5:  # drop songs that are shorter than all the others
            return file_path, False
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
    
    import json
    # dump valid files to dict from name to index, json
    valid_files_dict = {file_path: i for i, file_path in enumerate(valid_files)}
    with open(os.path.join(args.output_dir, 'name_to_index.json'), 'w') as f:
        json.dump(valid_files_dict, f)