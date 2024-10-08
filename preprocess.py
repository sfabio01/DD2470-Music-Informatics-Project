import librosa
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from typing import Tuple, List
from scipy.ndimage import zoom

def load_audio_mono(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and ensure it's mono using librosa."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def process_audio_file(file_path: str, output_dir: str) -> None:
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

        # Save processed data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(save_path, data)
        print(f"Processed and saved: {save_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_files_in_parallel(file_list: List[str], output_dir: str, num_workers: int = 4) -> None:
    """Process multiple audio files in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    with Pool(num_workers) as pool:
        func = partial(process_audio_file, output_dir=output_dir)
        pool.map(func, file_list)

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

    process_files_in_parallel(file_list, args.output_dir, num_workers=args.num_workers)

    end_time = time.time()
    print("Audio processing completed.")
    print(f"Total time taken: {end_time - start_time} seconds")
