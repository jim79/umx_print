from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2
        
        print(f'lstm_hidden_size {lstm_hidden_size}')

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        print(f'fc2_hiddensize {fc2_hiddensize}')

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

        # permute so that batch is last for lstm
        print('\n \n model forward \n \n')
        x = x.permute(3, 0, 1, 2)
        print(f'forward 1 : permute {x.shape}')
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        print('get current spectrogram shape')
        print(x.data.shape)
        print('nb_frames, nb_samples, nb_channels, nb_bins')

        mix = x.detach().clone()
        print(f'mix.shape {mix.shape}')

        # crop
        x = x[..., : self.nb_bins]
        print(f'forward 2 : crop {x.shape}')
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale
        print(f'forward 3 : scaler {x.shape}')

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        print(f'forward 4 : fc1 {x.shape}')
        print(f'nb_frames*nb_samples, nb_channels*nb_bins')

        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        print(f'forward 5: batch norm1 {x.shape}')
        print('nb_frames*nb_samples, hidden_size')

        # squash range ot [-1, 1]
        x = torch.tanh(x)
        print(f'forward 6: tanh {x.shape}')

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)
        print(f'forward 7: BLSTM {x.shape}')

        x = torch.cat([x, lstm_out[0]], -1)
        print(f'forward 8: Cat BLSTM out {x.shape}')


        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        print(f'forward 9 : first dense stage fc2 {x.shape}')

        x = self.bn2(x)
        print(f'forward 10 : batch norm 2 {x.shape}')

        x = F.relu(x)
        print(f'forward 11 : ReLu {x.shape}')

        # second dense stage + layer norm
        x = self.fc3(x)
        print(f'forward 12 : fsecond dense stage fc3 {x.shape}')

        x = self.bn3(x)
        print(f'forward 13 : batch norm 3 {x.shape}')

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        print(f' forward 14 : reshape back to original dim')
        print(x.shape)
        print('nb_frames, nb_samples, nb_channels, self.nb_output_bins')

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean
        print(f'forward 15 : output scaling {x.shape}')

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        print(f'forward 16 : ReLu {x.shape}')

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        print('permute back to (nb_samples, nb_channels, nb_bins, nb_frames')
        return x.permute(1, 2, 3, 0)



class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.istft(targets_stft, length=audio.shape[2])

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
