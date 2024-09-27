#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2024-09-27] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import onnxruntime
import numpy as np
import torch
import s3tokenizer


default = os.path.join(os.path.expanduser("~"), ".cache")
download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "s3tokenizer")
name = "speech_tokenizer_v1"
tokenizer = s3tokenizer.load_model(name)

mels = []
wav_paths = ["s3tokenizer/assets/BAC009S0764W0121.wav", "s3tokenizer/assets/BAC009S0764W0122.wav"]
for wav_path in wav_paths:
    audio = s3tokenizer.load_audio(wav_path)
    mels.append(s3tokenizer.log_mel_spectrogram(audio))
print("=========torch=============")
mels, mels_lens = s3tokenizer.padding(mels)
print(f"mels.size: {mels.size()}, mels_lens: {mels_lens}")
codes, codes_lens = tokenizer.quantize(mels, mels_lens)
print(f"codes.size: {codes.size()}, codes_lens: {codes_lens}")

for i in range(len(wav_paths)):
    print(f"wav[{i}]")
    print(codes[i, :codes_lens[i].item()])

print("=========onnx===============")
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
providers = ["CPUExecutionProvider"]
ort_session = onnxruntime.InferenceSession(f"{download_root}/{name}.onnx", sess_options=option, providers=providers)

for i in range(len(wav_paths)):
    speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: mels[i, :, :mels_lens[i].item()].unsqueeze(0).detach().cpu().numpy(),
                                          ort_session.get_inputs()[1].name: np.array([mels_lens[i].item()], dtype=np.int32)})[0]
    speech_token = torch.tensor(speech_token[0, 0, :])
    print(f"wav[{i}]")
    print(speech_token)
    print(f"all equal: {torch.equal(speech_token, codes[i, :codes_lens[i].item()].cpu())}")
    miss_num = torch.sum((speech_token == codes[i, :codes_lens[i].item()].cpu()) == False)
    total = speech_token.numel()
    print(f"miss rate: {miss_num * 100.0 / total}%")
