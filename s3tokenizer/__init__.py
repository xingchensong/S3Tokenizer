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
"""Modified from
    https://github.com/openai/whisper/blob/main/whisper/__init__.py
"""

import hashlib
import os
import urllib
import warnings
from typing import List, Union

from tqdm import tqdm

from s3tokenizer.model_v2 import S3TokenizerV2

from .model import S3Tokenizer
from .utils import (load_audio, log_mel_spectrogram, make_non_pad_mask,
                    mask_to_bias, onnx2torch, padding)

__all__ = [
    'load_audio', 'log_mel_spectrogram', 'make_non_pad_mask', 'mask_to_bias',
    'onnx2torch', 'padding'
]
_MODELS = {
    "speech_tokenizer_v1":
    "https://www.modelscope.cn/models/iic/cosyvoice-300m/"
    "resolve/master/speech_tokenizer_v1.onnx",
    "speech_tokenizer_v1_25hz":
    "https://www.modelscope.cn/models/iic/CosyVoice-300M-25Hz/"
    "resolve/master/speech_tokenizer_v1.onnx",
    "speech_tokenizer_v2_25hz":
    "https://www.modelscope.cn/models/iic/CosyVoice2-0.5B/"
    "resolve/master/speech_tokenizer_v2.onnx",
}

_SHA256S = {
    "speech_tokenizer_v1":
    "23b5a723ed9143aebfd9ffda14ac4c21231f31c35ef837b6a13bb9e5488abb1e",
    "speech_tokenizer_v1_25hz":
    "56285ddd4a83e883ee0cb9f8d69c1089b53a94b1f78ff7e4a0224a27eb4cb486",
    "speech_tokenizer_v2_25hz":
    "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71",
}


def _download(name: str, root: str) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = _SHA256S[name]
    url = _MODELS[name]
    download_target = os.path.join(root, f"{name}.onnx")

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not"
                " match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target,
                                                     "wb") as output:
        with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading onnx checkpoint",
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not"
            " match. Please retry loading the model.")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_model(
    name: str,
    download_root: str = None,
) -> S3Tokenizer:
    """
    Load a S3Tokenizer ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by
        `s3tokenizer.available_models()`, or path to a model checkpoint
         containing the model dimensions and the model state_dict.
    download_root: str
        path to download the model files; by default,
        it uses "~/.cache/s3tokenizer"

    Returns
    -------
    model : S3Tokenizer
        The S3Tokenizer model instance
    """

    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default),
                                     "s3tokenizer")

    if name in _MODELS:
        checkpoint_file = _download(name, download_root)
    elif os.path.isfile(name):
        checkpoint_file = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}")
    if 'v2' in name:
        model = S3TokenizerV2(name)
    else:
        model = S3Tokenizer(name)
    model.init_from_onnx(checkpoint_file)

    return model
