# Reverse Engineering of S3Tokenizer

<div align="center">
  <img src="https://arxiv.org/html/2407.04051v2/x1.png" alt="Description" width="35%" />
  <p><em>Supervised Semantic Speech Tokenizer (S3Tokenizer)</em></p>
</div>

S3Tokenizer was initially introduced in CosyVoice [[Paper]](https://arxiv.org/abs/2407.04051v2) [[Repo]](https://github.com/FunAudioLLM/CosyVoice), it is a Supervised Semantic Speech Tokenizer based on the pre-trained SenseVoice-Large model, which enhances the semantic relationship of extracted tokens to textual and paralinguistic information, is robust to data noise, and reduces the reliance on clean data collection, thereby enabling the use of a broader range of data for model training.

However, as indicated in this [[issue]](https://github.com/FunAudioLLM/CosyVoice/issues/70), the authors have no intention to open-source the PyTorch implementation of the S3Tokenizer, and only plan to release an ONNX file. Additionally, users aiming to fine-tune CosyVoice must extract speech codes offline, with the batch size restricted to 1, a process that is notably time-consuming (refer to [[cosyvoice/tools/extract_speech_token.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/tools/extract_speech_token.py)).

This repository undertakes a reverse engineering of the S3Tokenizer, offering:
1. A pure PyTorch implementation of S3Tokenizer (see [[model.py]](https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/model.py)), compatible with initializing weights from the released ONNX file (see [[utils.py::onnx2torch()]](https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/utils.py)).
2. High-throughput (distributed) batch inference, achieving a ~790x speedup compared to the original inference pipeline in [[cosyvoice/tools/extract_speech_token.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/tools/extract_speech_token.py).
3. The capability to perform online speech code extraction during SpeechLLM training.

## Latest News 🎉
- [2025/07/07] S3Tokenizer now has built-in **long audio processing** capabilities, requiring no additional operations from users!

## Supported Models 🔥
- [x] Model: [S3Tokenizer V1 50hz](https://modelscope.cn/models/iic/CosyVoice-300M)
- [x] Model: [S3Tokenizer V1 25hz](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)
- [x] Model: [S3Tokenizer V2 25hz](https://modelscope.cn/models/iic/CosyVoice2-0.5B)


# Setup

```sh
pip install s3tokenizer
```

# Usage-1: Offline batch inference

```py
import s3tokenizer

tokenizer = s3tokenizer.load_model("speech_tokenizer_v1").cuda()  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"

mels = []
wav_paths = ["s3tokenizer/assets/BAC009S0764W0121.wav", "s3tokenizer/assets/BAC009S0764W0122.wav"]
for wav_path in wav_paths:
    audio = s3tokenizer.load_audio(wav_path)
    mels.append(s3tokenizer.log_mel_spectrogram(audio))
mels, mels_lens = s3tokenizer.padding(mels)
codes, codes_lens = tokenizer.quantize(mels.cuda(), mels_lens.cuda())  # Automatically handles long audio internally!

for i in range(len(wav_paths)):
    print(codes[i, :codes_lens[i].item()])
```

# Usage-2: Distributed offline batch inference via command-line tools

## 2.1 CPU batch inference

```sh
s3tokenizer --wav_scp xxx.scp \
            --device "cpu" \
            --output_dir "./" \
            --batch_size 32 \
            --model "speech_tokenizer_v1"  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"
```



https://github.com/user-attachments/assets/d37d10fd-0e13-46a3-86b0-4cbec309086f



## 2.2 (Multi) GPU batch inference (a.k.a Distributed inference)

```sh
torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` --wav_scp xxx.scp \
                --device "cuda" \
                --output_dir "./" \
                --batch_size 32 \
                --model "speech_tokenizer_v1"  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"
```



https://github.com/user-attachments/assets/79a3fb11-7199-4ee2-8a35-9682a3b4d94a



## 2.3 Performance Benchmark

|  Method  | Time cost on Aishell Test Set | Relative speed up | Miss Rate |
|:------:|:----------:|:--------------:|:-----:|
|  [[cosyvoice/tools/extract_speech_token.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/tools/extract_speech_token.py), cpu |   9 hours    |    ~         | ~ |
|  cpu, batchsize 32  |    1.5h    |    ~6x        | 0.00% |
|  4 gpus (3090), batchsize 32 per gpu  |   41s    |   ~790x         | 0.00% |

The miss rate represents the proportion of tokens that are inconsistent between the batch inference predictions and the ONNX (batch=1) inference predictions.

# Usage-3: Online speech code extraction

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
        self.speech_tokenizer = s3tokenizer.load_model("speech_tokenizer_v1")  # or "speech_tokenizer_v1_25hz"
        self.speech_tokenizer.freeze()

    def forward(self, speech: Tensor, speech_lens: Tensor, text_ids: Tensor, ...):
        ...
        speech_codes, speech_codes_lens = self.speech_tokenizer.quantize(speech, speech_lens)
        speech_codes = speech_codes.clone()  # for backward compatbility
        speech_codes_lens = speeech_codes_lens.clone()  # for backward compatbility
```

</sub>
</td>
</tr>
</table>

# Usage-4: Long Audio Processing (Built-in Automatic Processing)

- **Automatic Detection**: Model automatically detects audio length (>30 seconds triggers long audio processing)
- **Sliding Window**: 30-second window with 4-second overlap, automatically segments long audio
- **Batch Processing**: Internal batch processing of multiple segments for improved efficiency
- **Complete Transparency**: User calling method is identical to short audio
