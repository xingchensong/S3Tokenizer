# Copyright (c)  (Mddct: Dinghao Zhou)
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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from s3tokenizer.model import Conv1d, LayerNorm, Linear
# Re-use V2 components where possible, but we might need specific V3 tweaks
from s3tokenizer.model_v2 import (FSMNMultiHeadAttention,
                                  FSQVectorQuantization, precompute_freqs_cis)
from s3tokenizer.utils import (make_non_pad_mask, mask_to_bias,
                               merge_tokenized_segments, onnx2torch_v3)


@dataclass
class ModelConfigV3:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 12  # V3 has 12 layers
    n_codebook_size: int = 3**8

    use_sdpa: bool = False


class MultiHeadAttentionV3(FSMNMultiHeadAttention):

    def __init__(self,
                 n_state: int,
                 n_head: int,
                 kernel_size: int = 31,
                 use_sdpa: bool = False):
        super().__init__(n_state, n_head, kernel_size, use_sdpa)
        # Override linears: query/value/out use bias=True, key uses bias=False
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)


class ResidualAttentionBlockV3(torch.nn.Module):

    def __init__(self,
                 n_state: int,
                 n_head: int,
                 kernel_size: int = 31,
                 use_sdpa: bool = False):
        super().__init__()
        self.attn = MultiHeadAttentionV3(n_state,
                                         n_head,
                                         kernel_size,
                                         use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state, eps=1e-5)
        n_mlp = n_state * 4
        # Set bias=True for MLP Linear layers
        self.mlp = torch.nn.Sequential(Linear(n_state, n_mlp), torch.nn.GELU(),
                                       Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state, eps=1e-5)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_pad: Optional[torch.Tensor] = None,
                freqs_cis: Optional[torch.Tensor] = None):
        x = x + self.attn(
            self.attn_ln(x), mask=mask, mask_pad=mask_pad,
            freqs_cis=freqs_cis)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV3(torch.nn.Module):

    def __init__(
        self,
        n_mels: int,
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
        self.freqs_cis = precompute_freqs_cis(64, 1024 * 2)
        # V3 uses the same ResidualAttentionBlock structure but more layers
        self.blocks = torch.nn.ModuleList([
            ResidualAttentionBlockV3(n_state, n_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        T = x.shape[-1]
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        x_slen = (T + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        x_slen = (x_slen + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[:x.size(1)])

        return x, x_len


class S3TokenizerV3(torch.nn.Module):
    """S3 tokenizer v3 implementation (inference-only).
    Args:
        config (ModelConfigV3): Config
    """

    def __init__(self, name: str, config: ModelConfigV3 = ModelConfigV3()):
        super().__init__()
        self.name = name
        self.config = config
        self.encoder = AudioEncoderV3(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def forward(self, mel: torch.Tensor,
                mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor,
                 mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Re-use logic from V2 (copy-paste or inheritance? Inheritance is trickier with imports)
        # Using exact same logic as V2 for now
        max_frames = 3000
        long_audio_mask = mel_len > max_frames

        if long_audio_mask.any():
            return self._quantize_mixed_batch(mel, mel_len, long_audio_mask)
        else:
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return code, code_len

    @torch.inference_mode()
    def _quantize_mixed_batch(
            self, mel: torch.Tensor, mel_len: torch.Tensor,
            long_audio_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = mel.size(0)
        sample_rate = 16000
        hop_length = 160
        window_size = 30
        overlap = 4

        frames_per_window = window_size * sample_rate // hop_length
        frames_per_overlap = overlap * sample_rate // hop_length
        frames_per_stride = frames_per_window - frames_per_overlap

        all_segments = []
        all_segments_len = []
        segment_info = []

        for batch_idx in range(batch_size):
            audio_mel = mel[batch_idx]
            audio_mel_len = mel_len[batch_idx]
            is_long_audio = long_audio_mask[batch_idx].item()

            if not is_long_audio:
                segment = audio_mel[:, :audio_mel_len]
                seg_len = audio_mel_len.item()
                if seg_len < frames_per_window:
                    pad_size = frames_per_window - seg_len
                    segment = torch.nn.functional.pad(segment, (0, pad_size))
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
                start = 0
                segment_idx = 0
                while start < audio_mel_len:
                    end = min(start + frames_per_window, audio_mel_len)
                    segment = audio_mel[:, start:end]
                    seg_len = segment.size(1)
                    if seg_len < frames_per_window:
                        pad_size = frames_per_window - seg_len
                        segment = torch.nn.functional.pad(
                            segment, (0, pad_size))
                    all_segments.append(segment)
                    all_segments_len.append(
                        torch.tensor(seg_len, device=mel.device))
                    segment_info.append({
                        'batch_idx': batch_idx,
                        'is_long_audio': True,
                        'segment_idx': segment_idx,
                        'total_segments': None
                    })
                    segment_idx += 1
                    # Stop once the current window covers the input tail.
                    if end >= audio_mel_len:
                        break
                    start += frames_per_stride
                total_segments = segment_idx
                for info in segment_info:
                    if info['batch_idx'] == batch_idx and info['is_long_audio']:
                        info['total_segments'] = total_segments

        if not all_segments:
            return torch.zeros(batch_size,
                               0,
                               dtype=torch.long,
                               device=mel.device), torch.zeros(
                                   batch_size,
                                   dtype=torch.long,
                                   device=mel.device)

        unified_batch_mel = torch.stack(all_segments)
        unified_batch_lens = torch.stack(all_segments_len)

        hidden, code_len = self.encoder(unified_batch_mel, unified_batch_lens)
        codes = self.quantizer.encode(hidden)

        results = {}

        for seg_idx, info in enumerate(segment_info):
            batch_idx = info['batch_idx']
            is_long_audio = info['is_long_audio']

            segment_code = codes[
                seg_idx, :code_len[seg_idx].item()].cpu().numpy().tolist()

            if not is_long_audio:
                code_tensor = torch.tensor(segment_code,
                                           dtype=torch.long,
                                           device=mel.device)
                results[batch_idx] = (code_tensor, len(segment_code))
            else:
                if batch_idx not in results:
                    results[batch_idx] = []
                results[batch_idx].append(segment_code)

        for batch_idx in range(batch_size):
            if long_audio_mask[batch_idx].item():
                audio_codes = results[batch_idx]
                token_rate = 25
                merged_codes = merge_tokenized_segments(audio_codes,
                                                        overlap=overlap,
                                                        token_rate=token_rate)
                merged_codes_tensor = torch.tensor(merged_codes,
                                                   dtype=torch.long,
                                                   device=mel.device)
                results[batch_idx] = (merged_codes_tensor, len(merged_codes))

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
        ckpt = onnx2torch_v3(onnx_path, None,
                             False)  # Set verbose back to False
        self.load_state_dict(ckpt, strict=False)

    def init_from_pt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        self.load_state_dict(ckpt, strict=True)

    def load_state_dict(self, state_dict, strict=True):
        # Allow loading state dict with missing keys (like LN bias) if strictly necessary
        # But for now we try to stick to standard behavior
        return super().load_state_dict(state_dict, strict=strict)

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
