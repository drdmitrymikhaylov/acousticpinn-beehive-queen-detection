"""Log-mel front end and a small convolutional classifier.

The filterbank is written out from the STFT rather than imported, so the whole
front end is inspectable and the low band -- where hive sound actually lives --
is not quietly discarded by a default fmin.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SR = 8000
N_FFT = 1024
HOP = 256
N_MELS = 64
FMIN, FMAX = 20.0, 3800.0


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
        fb[i] *= 2.0 / max(hi - lo, 1e-9)
    return torch.from_numpy(fb)


class LogMel(nn.Module):
    def __init__(self, fmin=FMIN):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("fb", mel_filterbank(fmin=fmin))

    def forward(self, x):
        spec = torch.stft(x, N_FFT, HOP, window=self.window,
                          return_complex=True, center=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.log(torch.matmul(self.fb, power) + 1e-8)
        mu = mel.mean(dim=(-2, -1), keepdim=True)
        sd = mel.std(dim=(-2, -1), keepdim=True) + 1e-5
        return (mel - mu) / sd


class BeeNet(nn.Module):
    def __init__(self, n_classes=2, fmin=FMIN):
        super().__init__()
        self.front = LogMel(fmin=fmin)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))

        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.body(self.front(x).unsqueeze(1)))
