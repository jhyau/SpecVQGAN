import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random


def files_to_list(filename, data_root):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    mel_files = [data_root / f'{f.rstrip()}_mel.npy' for f in files]
    files = [data_root / f'{f.rstrip()}_audio.npy' for f in files]
    return files, mel_files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, data_root, split_path, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files, self.mel_files = files_to_list(split_path, data_root)
        # self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        mel_spec_filename = self.mel_files[index]
        #print(f"audio file name: {filename} and mel spec file name: {mel_spec_filename}")
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0

        # Get the corresponding mel spectrogram
        mel_spec = np.load(mel_spec_filename)

        return audio.unsqueeze(0), mel_spec

    def __len__(self):
        assert(len(self.audio_files) == len(self.mel_files))
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data = np.load(full_path)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), self.sampling_rate
