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
"""Modified from https://github.com/openai/whisper/blob/main/whisper/model.py
   Add EuclideanCodebook & VectorQuantization
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .utils import make_non_pad_mask, mask_to_bias, onnx2torch, merge_tokenized_segments


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 4096

    use_sdpa: bool = False


class LayerNorm(nn.LayerNorm):

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
        np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, use_sdpa: bool = False):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.use_sdpa = use_sdpa

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self,
                      q: Tensor,
                      k: Tensor,
                      v: Tensor,
                      mask: Optional[Tensor] = None):
        _, _, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1,
                                   3).flatten(start_dim=2), qk.detach()
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.,
                scale=1.,
            )
            output = (output.transpose(1,
                                       2).contiguous().view(q.size(0), -1, D)
                      )  # (batch, time1, d_model)
            return output, None


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: int, n_head: int, use_sdpa: bool):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(),
                                 Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv1d(n_mels,
                            n_state,
                            kernel_size=3,
                            stride=stride,
                            padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)

        mask = mask_to_bias(mask, x.dtype)

        x = (x + self.positional_embedding[:x.shape[1], :]).to(x.dtype)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))

        return x, x_len


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance (inference-only).
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        embed = torch.zeros(codebook_size, dim)
        self.codebook_size = codebook_size
        self.register_buffer("embed", embed)

    @torch.inference_mode()
    def preprocess(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def quantize(self, x: Tensor) -> Tensor:
        embed = self.embed.t().to(x.dtype)
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                 embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    @torch.inference_mode()
    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    @torch.inference_mode()
    def dequantize(self, embed_ind: Tensor) -> Tensor:
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self.dequantize(embed_ind)
        return quantize


class VectorQuantization(nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._codebook = EuclideanCodebook(dim=dim,
                                           codebook_size=codebook_size)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        x = F.normalize(x.float(), p=2, dim=-1)
        embed_in = self._codebook.encode(x)
        return embed_in

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class S3Tokenizer(nn.Module):
    """S3 tokenizer implementation (inference-only).
    Args:
        config  (ModelConfig): Config
    """

    def __init__(self, name: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.name = name  # Store model name for token_rate determination
        self.config = config
        self.encoder = AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2 if name == "speech_tokenizer_v1_25hz" else 1,
            self.config.use_sdpa,
        )
        self.quantizer = VectorQuantization(self.config.n_audio_state,
                                            self.config.n_codebook_size)

    def forward(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize mel spectrogram to tokens, with automatic long audio handling.

        Args:
            mel: mel spectrogram tensor, shape (batch_size, n_mels, T)
            mel_len: mel length tensor, shape (batch_size,)

        Returns:
            code: quantized tokens, shape (batch_size, T')
            code_len: token length, shape (batch_size,)
        """
        # Check if any audio in the batch exceeds 30 seconds
        # Assuming 16kHz sample rate and hop_length=160, 30s = 30*16000/160 = 3000 frames
        max_frames = 3000

        # Check which samples are long audio
        long_audio_mask = mel_len > max_frames

        if long_audio_mask.any():
            # Has long audio - need special processing
            return self._quantize_mixed_batch(mel, mel_len, long_audio_mask,
                                              max_frames)
        else:
            # All short audio - use original method
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return code, code_len

    @torch.inference_mode()
    def _quantize_mixed_batch(self, mel: Tensor, mel_len: Tensor,
                              long_audio_mask: Tensor,
                              max_frames: int) -> Tuple[Tensor, Tensor]:
        """
        Handle mixed batch with both short and long audio using unified batch processing.

        Args:
            mel: mel spectrogram tensor, shape (batch_size, n_mels, T)
            mel_len: mel length tensor, shape (batch_size,)
            long_audio_mask: boolean mask for long audio, shape (batch_size,)
            max_frames: maximum frames for short audio

        Returns:
            code: quantized tokens, shape (batch_size, T')
            code_len: token length, shape (batch_size,)
        """
        batch_size = mel.size(0)

        # Parameters for sliding window
        sample_rate = 16000
        hop_length = 160  # Default hop length for mel spectrogram
        window_size = 30  # seconds
        overlap = 4  # seconds

        # Calculate frame-based parameters
        frames_per_window = window_size * sample_rate // hop_length  # 3000 frames
        frames_per_overlap = overlap * sample_rate // hop_length  # 400 frames
        frames_per_stride = frames_per_window - frames_per_overlap  # 2600 frames

        # Collect all segments to process (including short and long audio segments)
        all_segments = []
        all_segments_len = []
        segment_info = [
        ]  # Record which audio each segment belongs to and whether it's long audio

        # Process all audio in the batch
        for batch_idx in range(batch_size):
            audio_mel = mel[batch_idx]
            audio_mel_len = mel_len[batch_idx]
            is_long_audio = long_audio_mask[batch_idx].item()

            if not is_long_audio:
                # Short audio: process directly as a single segment
                segment = audio_mel[:, :audio_mel_len]
                seg_len = audio_mel_len.item()

                # Pad to max_frames if necessary
                if seg_len < frames_per_window:
                    pad_size = frames_per_window - seg_len
                    segment = F.pad(segment, (0, pad_size))

                all_segments.append(segment)
                all_segments_len.append(
                    torch.tensor(seg_len, device=mel.device))
                segment_info.append({
                    'batch_idx': batch_idx,
                    'is_long_audio': False,
                    'segment_idx': 0,
                    'total_segments': 1
                })
            else:
                # Long audio: split into multiple segments
                start = 0
                segment_idx = 0
                while start < audio_mel_len:
                    end = min(start + frames_per_window, audio_mel_len)
                    segment = audio_mel[:, start:end]

                    seg_len = segment.size(1)
                    # Pad if necessary
                    if seg_len < frames_per_window:
                        pad_size = frames_per_window - seg_len
                        segment = F.pad(segment, (0, pad_size))

                    all_segments.append(segment)
                    all_segments_len.append(
                        torch.tensor(seg_len, device=mel.device))
                    segment_info.append({
                        'batch_idx': batch_idx,
                        'is_long_audio': True,
                        'segment_idx': segment_idx,
                        'total_segments': None  # Will be filled later
                    })

                    segment_idx += 1
                    start += frames_per_stride

                # Update total_segments info
                total_segments = segment_idx
                for info in segment_info:
                    if info['batch_idx'] == batch_idx and info['is_long_audio']:
                        info['total_segments'] = total_segments

        if not all_segments:
            # Fallback if no segments
            return torch.zeros(batch_size,
                               0,
                               dtype=torch.long,
                               device=mel.device), torch.zeros(
                                   batch_size,
                                   dtype=torch.long,
                                   device=mel.device)

        # Unified batch processing for all segments
        unified_batch_mel = torch.stack(all_segments)
        unified_batch_lens = torch.stack(all_segments_len)

        # Process all segments at once
        hidden, code_len = self.encoder(unified_batch_mel, unified_batch_lens)
        codes = self.quantizer.encode(hidden)

        # Reorganize results based on segment_info
        results = {}  # batch_idx -> (code_tensor, code_len)

        for seg_idx, info in enumerate(segment_info):
            batch_idx = info['batch_idx']
            is_long_audio = info['is_long_audio']
            segment_idx = info['segment_idx']

            # Get codes for current segment
            segment_code = codes[
                seg_idx, :code_len[seg_idx].item()].cpu().numpy().tolist()

            if not is_long_audio:
                # Short audio: use directly
                code_tensor = torch.tensor(segment_code,
                                           dtype=torch.long,
                                           device=mel.device)
                results[batch_idx] = (code_tensor, len(segment_code))
            else:
                # Long audio: collect all segments
                if batch_idx not in results:
                    results[batch_idx] = []
                results[batch_idx].append(segment_code)

        # Process long audio segment merging
        for batch_idx in range(batch_size):
            if long_audio_mask[batch_idx].item():
                # Merge long audio segments
                audio_codes = results[batch_idx]

                # Determine token rate based on model name
                if hasattr(self,
                           'name') and self.name == "speech_tokenizer_v1":
                    token_rate = 50
                else:
                    token_rate = 25

                merged_codes = merge_tokenized_segments(audio_codes,
                                                        overlap=overlap,
                                                        token_rate=token_rate)

                # Convert to tensor
                merged_codes_tensor = torch.tensor(merged_codes,
                                                   dtype=torch.long,
                                                   device=mel.device)
                results[batch_idx] = (merged_codes_tensor, len(merged_codes))

        # Construct final output
        max_code_len = max(code_info[1] for code_info in results.values())

        output_codes = torch.zeros(batch_size,
                                   max_code_len,
                                   dtype=torch.long,
                                   device=mel.device)
        output_codes_len = torch.zeros(batch_size,
                                       dtype=torch.long,
                                       device=mel.device)

        for batch_idx, (code_tensor, code_len) in results.items():
            output_codes[batch_idx, :code_len] = code_tensor
            output_codes_len[batch_idx] = code_len

        return output_codes, output_codes_len

    @property
    def device(self):
        return next(self.parameters()).device

    def init_from_onnx(self, onnx_path: str):
        ckpt = onnx2torch(onnx_path, None, False)
        self.load_state_dict(ckpt, strict=True)

    def init_from_pt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        self.load_state_dict(ckpt, strict=True)

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
