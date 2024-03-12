import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

rootDir = 'data'  
targetDir_f0 = 'f0_dir'
targetDir = 'spmel_dir'

print('Root directory:', rootDir)
print('Target directory for f0:', targetDir_f0)
print('Target directory for spmel:', targetDir)

speaker_dirs = [d for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, d))]
print('Found speaker directories:', speaker_dirs)

mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

for speaker_dir in speaker_dirs:
    print('Processing speaker directory:', speaker_dir)
    
    speaker_wav_dir = os.path.join(rootDir, speaker_dir, 'wav')
    if not os.path.exists(speaker_wav_dir):
        print('No "wav" directory found for speaker:', speaker_dir)
        continue
    
    if not os.path.exists(os.path.join(targetDir, speaker_dir)):
        os.makedirs(os.path.join(targetDir, speaker_dir))
        print('Created directory for spmel:', os.path.join(targetDir, speaker_dir))
    if not os.path.exists(os.path.join(targetDir_f0, speaker_dir)):
        os.makedirs(os.path.join(targetDir_f0, speaker_dir))
        print('Created directory for f0:', os.path.join(targetDir_f0, speaker_dir))
    
    wav_files = [f for f in os.listdir(speaker_wav_dir) if f.endswith('.wav')]
    print('Found {} wav files for speaker: {}'.format(len(wav_files), speaker_dir))
    
    lo, hi = 100, 600  
    
    prng = RandomState(np.random.randint(0, 1000))  
    for wav_file in sorted(wav_files):
        print('Processing wav file:', wav_file)
        
        # read audio file
        x, fs = sf.read(os.path.join(speaker_wav_dir, wav_file))
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        
        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100        
        
        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, speaker_dir, wav_file[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        print('Saved spmel feature:', os.path.join(targetDir, speaker_dir, wav_file[:-4] + '.npy'))
        
        np.save(os.path.join(targetDir_f0, speaker_dir, wav_file[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)
        print('Saved f0 feature:', os.path.join(targetDir_f0, speaker_dir, wav_file[:-4] + '.npy'))

print('Processing completed.')
