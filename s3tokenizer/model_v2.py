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
from einops import rearrange

from s3tokenizer.model import Conv1d, LayerNorm, Linear, MultiHeadAttention
from s3tokenizer.utils import make_non_pad_mask, mask_to_bias, onnx2torch


@dataclass
class ModelDimensions:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         scaling=None):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D // 2], xq[:, :, :, D // 2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]

    half_l, half_r = xk[:, :, :, :D // 2], xk[:, :, :, D // 2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


class FSQCodebook(torch.nn.Module):

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize

        h = self.project_down(x)
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        # h = ((self.level - 1) * h).round()  # range [-k, k]
        powers = torch.pow(self.level,
                           torch.arange(2**self.level, device=x.device))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1]).int()
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'There is no official up project component provided')


class FSQVectorQuantization(torch.nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__(n_state, n_head)

        self.fsmn_block = torch.nn.Conv1d(n_state,
                                          n_state,
                                          kernel_size,
                                          stride=1,
                                          padding=0,
                                          groups=n_state,
                                          bias=False)
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d(
            (self.left_padding, self.right_padding), 0.0)

    def forward_fsmn(self,
                     inputs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:  # time2 > 0
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(self,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      mask_pad: Optional[torch.Tensor] = None,
                      freqs_cis: Optional[torch.Tensor] = None):
        _, _, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        k = k.permute(0, 2, 3, 1) * scale
        v = v.permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()
        w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1,
                               3).flatten(start_dim=2), qk.detach(), fsm_memory

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_pad: Optional[torch.Tensor] = None,
                freqs_cis: Optional[torch.Tensor] = None):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad,
                                                freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(torch.nn.Module):

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size)
        self.attn_ln = LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4

        self.mlp = torch.nn.Sequential(Linear(n_state, n_mlp), torch.nn.GELU(),
                                       Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(
            self.attn_ln(x), mask=mask, mask_pad=mask_pad,
            freqs_cis=freqs_cis)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(torch.nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
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
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        T = x.size(-1)
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        freqs_cis = self.freqs_cis.to(x.device)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask[:, :, (T + 2) % 2::2]  # (B, 1, T // 2)
        mask_pad = None
        if self.stride == 2:
            _T = mask.size(-1)
            mask = mask[:, :, (_T + 1) % 2::2]  # (B, 1, T // 4)
            mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)

        tmp = torch.view_as_real(freqs_cis)
        cos, sin = tmp[:, :, 0], tmp[:, :, 1]

        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[:x.size(1)])

        x_len = (x_len + 1) // 2
        if self.stride == 2:
            x_len = (x_len + 1) // 2
        return x, x_len


class S3TokenizerV2(torch.nn.Module):
    """S3 tokenizer v2 implementation (inference-only).
    Args:
        dims (ModelDimensions): Dimension
    """

    def __init__(self, name: str, dims: ModelDimensions = ModelDimensions()):
        super().__init__()
        if 'v1' not in name:
            assert 'v2' in name
            # TODO(Mddct): make it configureable
            dims.n_codebook_size = 3**8
        self.dims = dims
        self.encoder = AudioEncoderV2(
            self.dims.n_mels,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            2,
        )
        self.quantizer = FSQVectorQuantization(
            self.dims.n_audio_state,
            self.dims.n_codebook_size,
        )

    def forward(self, mel: torch.Tensor,
                mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor,
                 mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len

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
