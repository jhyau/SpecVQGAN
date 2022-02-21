import scipy.io.wavfile
from mel2wav.extract_mel_spectrogram import TRANSFORMS
import torch
import librosa

def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)

def wav2mel(batch, wave_len=None):

    if len(batch.shape) == 3:
        assert batch.shape[1] == 1, 'Multi-channel audio?'
        batch = batch.squeeze(1)

    batch = torch.stack([torch.from_numpy(TRANSFORMS(e.numpy())) for e in batch.cpu()]).float()

    if wave_len is not None:
        batch = batch[:, :, :wave_len]

    return batch

def wave2melWaveglow(batch, wave_len=None):
    if len(batch.shape) == 3:
        assert batch.shape[1] == 1, 'Multi-channel audio?'
        batch = batch.squeeze(1)

    #batch = torch.stack([torch.from_numpy(TRANSFORMS(e.numpy())) for e in batch.cpu()]).float()
    MAX_WAV_VALUE = 32768.0

    # Get the mel spectrogram
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    # Set up STFT to get mel spectrograms
    #stft = TacotronSTFT(filter_length=args.filter_length, hop_length=args.hop_length, win_length=args.win_length, sampling_rate=args.sampling_rate, mel_fmin=args.fmin, mel_fmax=args.fmax)
    #melspec = stft.mel_spectrogram(audio_norm)
    mel_basis = librosa.filters.mel(sr=44100, n_fft=1024, fmin=0.0, fmax=16000.0, n_mels=80)
    melspec = librosa.stft(audio_norm, n_fft=1024, hop_length=256, win_length=1024)
    melspec = np.dot(mel_basis, melspec)
    melspec = torch.squeeze(melspec, 0)
    #print(f'Output melspec shape original: {melspec.size()}')
    melspec = melspec[:, :1720]
    melspec = melspec.detach().cpu().numpy() # Convert torch tensor to numpy

    if wave_len is not None:
        batch = batch[:, :, :wave_len]

    return batch
