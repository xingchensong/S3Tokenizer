# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py
   Add rename_weights() & onnx2torch() & make_non_pad_mask() & mask_to_bias()
   Copy merge_tokenized_segments() from https://github.com/Mddct/s3tokenizer-long/blob/main/example.py
"""

import os
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def _rename_weights(weights_dict: dict):
    """
    Rename onnx weights to pytorch format.

    Parameters
    ----------
    weight_dict: dict
        The dict containing weights in onnx format

    Returns
    -------
    A new weight dict containing the weights in pytorch format.
    """
    new_weight_dict = {}
    for k in weights_dict.keys():
        if "quantizer" in k:  # vq or fsq
            if k == "/quantizer/rq/model/layers.0/_codebook/Pow_1":
                new_weight_dict["quantizer._codebook.embed"] = weights_dict[k]
            elif 'project_down' in k:  # v2
                new_weight_dict[k] = weights_dict[k]
        elif "positional_embedding" in k:  # positional emb
            new_weight_dict[k] = weights_dict[k]
        elif "conv" in k:  # 1/2 or 1/4 subsample
            new_weight_dict[k] = weights_dict[k]
        else:  # transformer blocks
            assert "blocks" in k
            new_k = (k[1:].replace('/', '.').replace(
                'MatMul', 'weight').replace('Add_1', 'bias').replace(
                    'Mul', 'weight').replace('Add', 'bias').replace(
                        'mlp.mlp', 'mlp')).replace('fsmn_block.Conv',
                                                   'fsmn_block.weight')

            new_weight_dict[f"encoder.{new_k}"] = weights_dict[k]
    return new_weight_dict


def onnx2torch(onnx_path: str, torch_path: str = None, verbose: bool = False):
    """
    Open an onnx file and convert to pytorch format.

    Parameters
    ----------
    onnx_path: str
        The onnx file to open, typically `speech_tokenizer_v1.onnx`

    torch_path: str
        The path to save the torch-formated checkpoint.

    verbose: bool
        Logging info or not.

    Returns
    -------
    A checkpoint dict containing the weights and their names, if torch_path is
    None. Otherwise save checkpoint dict to the desired path.
    """
    onnx_model = onnx.load(onnx_path)
    weights_dict = {}
    initializer_map = {
        initializer.name: initializer
        for initializer in onnx_model.graph.initializer
    }
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_map:
                ln_bias_name, ln_weight_name = None, None  # for v2 ln
                initializer = initializer_map[input_name]
                if input_name in [
                        "onnx::Conv_1519",
                        "encoders.conv1.weight",
                        "onnx::Conv_2216",
                ]:  # v1_50hz, v1_25hz, v2_25hz
                    weight_name = "encoder.conv1.weight"
                elif input_name in [
                        "onnx::Conv_1520",
                        "encoders.conv1.bias",
                        "onnx::Conv_2217",
                ]:  # v1_50hz, v1_25hz, v2_25hz
                    weight_name = "encoder.conv1.bias"
                elif input_name in [
                        "onnx::Conv_1521",
                        "encoders.conv2.weight",
                        "onnx::Conv_2218",
                ]:
                    weight_name = "encoder.conv2.weight"
                elif input_name in [
                        "onnx::Conv_1522",
                        "encoders.conv2.bias",
                        "onnx::Conv_2219",
                ]:
                    weight_name = "encoder.conv2.bias"
                elif input_name == "encoders.positional_embedding":
                    weight_name = "encoder.positional_embedding"
                elif input_name == 'quantizer.project_in.bias':
                    weight_name = "quantizer._codebook.project_down.bias"
                elif input_name == 'onnx::MatMul_2536':
                    weight_name = "quantizer._codebook.project_down.weight"
                else:
                    if node.op_type == 'LayerNormalization':  # in input_name:
                        ln_name = node.name.replace('/LayerNormalization', '')
                        ln_weight_name = ln_name + '.weight'
                        ln_bias_name = ln_name + '.bias'
                    else:
                        weight_name = node.name
                if ln_weight_name is not None and ln_bias_name is not None:
                    ln_inputs = node.input
                    scale_name = ln_inputs[1]
                    bias_name = ln_inputs[2]
                    scale = onnx.numpy_helper.to_array(
                        initializer_map[scale_name]).copy(
                        ) if scale_name in initializer_map else None
                    bias = onnx.numpy_helper.to_array(
                        initializer_map[bias_name]).copy(
                        ) if bias_name in initializer_map else None
                    scale.flags.writeable = True
                    bias.flags.writeable = True
                    weight_tensor = torch.from_numpy(scale)
                    bias_tensor = torch.from_numpy(bias)

                    weights_dict[ln_bias_name] = bias_tensor
                    weights_dict[ln_weight_name] = weight_tensor
                else:
                    weight_array = onnx.numpy_helper.to_array(
                        initializer).copy()
                    weight_array.flags.writeable = True
                    weight_tensor = torch.from_numpy(weight_array)
                    if len(weight_tensor.shape) > 2 or weight_name in [
                            "encoder.positional_embedding"
                    ]:
                        weights_dict[weight_name] = weight_tensor
                    else:
                        weights_dict[weight_name] = weight_tensor.t()

    new_weights_dict = _rename_weights(weights_dict)
    if verbose:
        for k, v in new_weights_dict.items():
            print(f"{k} : {v.shape} {v.dtype}")
        print(f"PyTorch weights saved to {torch_path}")
    del weights_dict, onnx_model
    if torch_path:
        torch.save(new_weights_dict, torch_path)
    else:
        return new_weights_dict


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A torch.Tensor containing the audio waveform, in float32 dtype.
    """
    audio, sample_rate = torchaudio.load(file)
    if sample_rate != sr:
        audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
    audio = audio[0]  # get the first channel
    return audio


@lru_cache(maxsize=None)
def _mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets",
                                "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the
        audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (128, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = _mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks =
            [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
             [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
             [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)

    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def padding(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats [B, 128, T_max], feats lengths [B]
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample],
                                 dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


def merge_tokenized_segments(tokenized_segments, overlap, token_rate):
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
    - tokenized_segments (List[List[int]]): List of tokenized sequences.
    - overlap (int): Overlapping duration in seconds (default: 4s).
    - token_rate (int): Number of tokens per second.

    Returns:
    - List[int]: A single merged token sequence.
    """
    merged_tokens = []
    overlap_tokens = (
        overlap //
        2) * token_rate  # Tokens corresponding to half of the overlap duration

    for i, tokens in enumerate(tokenized_segments):
        l = 0 if i == 0 else overlap_tokens
        r = -overlap_tokens if i != len(tokenized_segments) - 1 else len(tokens)
        # Keep only the middle part (drop overlap / 2 from both sides)
        merged_tokens.extend(tokens[l:r])

    return merged_tokens
