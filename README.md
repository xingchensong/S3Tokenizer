# Reverse Engineering of S3Tokenizer

<div align="center">
  <img src="https://arxiv.org/html/2407.04051v2/x1.png" alt="Description" width="35%" />
  <p><em>Supervised Semantic Speech Tokenizer (S3Tokenizer)</em></p>
</div>

S3Tokenizer was initially introduced in CosyVoice [[Paper]](https://arxiv.org/abs/2407.04051v2) [[Repo]](https://github.com/FunAudioLLM/CosyVoice), it is a Supervised Semantic Speech Tokenizer based on the pre-trained SenseVoice-Large model, which enhances the semantic relationship of extracted tokens to textual and paralinguistic information, is robust to data noise, and reduces the reliance on clean data collection, thereby enabling the use of a broader range of data for model training.

However, as indicated in this [[issue]](https://github.com/FunAudioLLM/CosyVoice/issues/70), the authors have no intention to open-source the PyTorch implementation of the S3Tokenizer, and only plan to release an ONNX file. Additionally, users aiming to fine-tune CosyVoice must extract speech codes offline, with the batch size restricted to 1, a process that is notably time-consuming (refer to [[cosyvoice/tools/extract_speech_token.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/tools/extract_speech_token.py)).

This repository undertakes a reverse engineering of the S3Tokenizer, offering:
1. A pure PyTorch implementation of S3Tokenizer, compatible with initializing weights from the released ONNX file.
2. High-throughput batch inference, achieving a 30x speedup compared to the original inference pipeline in [[cosyvoice/tools/extract_speech_token.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/tools/extract_speech_token.py).
3. The capability to perform online speech code extraction during SpeechLLM training.

# Setup

```sh
pip install s3tokenizer
```

# Usage-1: Offline batch inference

```py
import torch
import s3tokenizer

tokenizer = s3tokenizer.load_model("speech_tokenizer_v1").cuda()

mels, mels_lens = [], []
wav_paths = ["path_to_wav1", "path_to_wav2", ... "path_to_wavn"]
for wav_path in wav_paths:
    audio = s3tokenizer.load_audio(wav_path)
    mels.append(s3tokenizer.log_mel_spectrogram(audio))
mels, mels_lens = s3tokenizer.padding(mels)
codes, codes_lens = tokenizer.quantize(mels.cuda(), mels_lens.cuda())

for i in range(len(wav_paths)):
    print(codes[i, :codes_lens[i].item()])
```


# Usage-2: Online speech code extraction (TODO)

<table>
<tr>
<th>Before (extract code offline)</th>
<th>After (extract code online)</th>
</tr>
<tr>
<td>
<sub>

```py

class SpeechLLM(nn.Module):
    ...
    def __init__(self, ...):
        ...

    def forward(self, speech_codes: Tensor, text_ids: Tensor, ...):
        ...
```

</sub>
<td>
<sub>

```py
import s3tokenizer

class SpeechLLM(nn.Module):
    ...
    def __init__(self, ...):
        ...
        self.speech_tokenizer = s3tokenizer.load_model("speech_tokenizer_v1")

    def forward(self, speech: Tensor, speech_lens: Tensor, text_ids: Tensor, ...):
        ...
        speech_codes = self.speech_tokenizer(speech, speech_lens)
```

</sub>
</td>
</tr>
</table>

# Usage-3: Command-line (TODO)

```sh
s3tokenizer --wav_scp "xxx.scp" --device "cuda:0" --output "yyy.list" --batch_size 32
```
