
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

"""
Ref: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/feature.py#L26
Ref2: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
Ref3: https://github.com/zyzisyz/mfa_conformer/blob/master/module/feature.py
"""

    


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=128, n_mels=128, **kwargs):
        super(Mel_Spectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, \
                                                                    n_fft=self.n_fft,\
                                                                    win_length=self.win_length,\
                                                                    hop_length=self.hop_length, \
                                                                    n_mels=self.n_mels, \
                                                                    f_min = 20, f_max = 7600, \
                                                                    window_fn=torch.hamming_window, )


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch, time)
        Returns:
            x (torch.Tensor): (batch, n_mels, time)
        """
        with torch.no_grad():
            x = self.mel_spectrogram(x) + 1e-6
            x = torch.log(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)

        return x
    

def feature_extractor(sample_rate=16000, n_fft=512, win_length=512, hop_length=128, n_mels=128, **kwargs):
    return Mel_Spectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, **kwargs)