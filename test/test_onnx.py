#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2024-09-27] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import time
from typing import Dict, Any

import numpy as np
import onnxruntime
import pytest
import s3tokenizer
import torch


def create_test_audio(duration_seconds: float = 20,
                      sample_rate: int = 16000) -> torch.Tensor:
    """Create synthetic test audio"""
    length = int(duration_seconds * sample_rate)
    # Create sinusoidal mixed audio
    t = torch.linspace(0, duration_seconds, length)
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)  # 440Hz fundamental
    audio += 0.3 * torch.sin(2 * torch.pi * 880 * t)  # 880Hz second harmonic
    audio += 0.1 * torch.randn(length)  # Add noise
    return audio


@pytest.fixture
def test_audio_suite():
    """Create a suite of test audios with different lengths"""
    return {
        "short_audio_1": create_test_audio(5.0),  # 5 seconds
        "short_audio_2": create_test_audio(15.0),  # 15 seconds
        "medium_audio": create_test_audio(25.0),  # 25 seconds
        "medium_audio_2": create_test_audio(30.0),  # 30 seconds
        "long_audio": create_test_audio(
            35.0),  # 35 seconds - for torch and onnx, 2 segments with padding
        "long_audio_2": create_test_audio(
            56.0
        ),  # 56 seconds - for torch and onnx, exactly 2 segments without padding
        "very_long_audio": create_test_audio(
            60.0),  # 60 seconds - for torch and onnx, 3 segments with padding
    }


def onnx_inference_short_audio(model_name: str, mel: torch.Tensor,
                               mel_len: torch.Tensor) -> torch.Tensor:
    """
    ONNX inference for short audio (<=30s)
    """
    # Load ONNX model
    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default),
                                 "s3tokenizer")

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]

    ort_session = onnxruntime.InferenceSession(
        f"{download_root}/{model_name}.onnx",
        sess_options=option,
        providers=providers)

    # Direct inference for short audio
    onnx_output = ort_session.run(
        None, {
            ort_session.get_inputs()[0].name:
            mel[:, :mel_len.item()].unsqueeze(0).detach().cpu().numpy(),
            ort_session.get_inputs()[1].name:
            np.array([mel_len.item()], dtype=np.int32)
        })[0]

    # Convert to numpy array to fix linter issues
    onnx_output = np.array(onnx_output)

    # Handle different output formats
    if onnx_output.ndim == 2:
        onnx_output = onnx_output[0, :]
    elif onnx_output.ndim == 3:
        onnx_output = onnx_output[0, 0, :]

    return torch.tensor(onnx_output, dtype=torch.long)


def onnx_inference_long_audio(model_name: str, mel: torch.Tensor,
                              mel_len: torch.Tensor) -> torch.Tensor:
    """
    ONNX inference for long audio (>30s) using sliding window approach
    Based on _quantize_mixed_batch logic

    Note: This may fail due to ONNX model limitations with dynamic lengths
    """
    # Load ONNX model
    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default),
                                 "s3tokenizer")

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]

    ort_session = onnxruntime.InferenceSession(
        f"{download_root}/{model_name}.onnx",
        sess_options=option,
        providers=providers)

    # Parameters for sliding window (same as _quantize_mixed_batch)
    sample_rate = 16000
    hop_length = 160
    window_size = 30  # seconds
    overlap = 4  # seconds

    # Calculate frame-based parameters
    frames_per_window = window_size * sample_rate // hop_length  # 3000 frames
    frames_per_overlap = overlap * sample_rate // hop_length  # 400 frames
    frames_per_stride = frames_per_window - frames_per_overlap  # 2600 frames

    # Split into segments
    segments = []
    segments_len = []
    start = 0

    while start < mel_len.item():
        end = min(start + frames_per_window, mel_len.item())
        segment = mel[:, start:end]

        if segment.size(1) < frames_per_window:
            break

        seg_len = segment.size(1)
        segments.append(segment)
        segments_len.append(seg_len)

        start += frames_per_stride

    if not segments:
        raise ValueError("No valid segments for ONNX processing")

    # Process each segment with ONNX
    segment_results = []
    for i, (segment, seg_len) in enumerate(zip(segments, segments_len)):
        try:
            onnx_output = ort_session.run(
                None, {
                    ort_session.get_inputs()[0].name:
                    segment.unsqueeze(0).detach().cpu().numpy(),
                    ort_session.get_inputs()[1].name:
                    np.array([seg_len], dtype=np.int32)
                })[0]

            # Convert to numpy array to fix linter issues
            onnx_output = np.array(onnx_output)

            # Handle different output formats
            if onnx_output.ndim == 2:
                segment_codes = onnx_output[0, :].tolist()
            elif onnx_output.ndim == 3:
                segment_codes = onnx_output[0, 0, :].tolist()
            else:
                segment_codes = onnx_output.tolist()

            segment_results.append(segment_codes)

        except Exception as e:
            print(f"  ONNX error on segment {i+1}: {str(e)[:100]}...")
            raise Exception(
                f"ONNX inference failed on segment {i+1}: {str(e)}")

    if not segment_results:
        raise ValueError("All ONNX segments failed to process")

    # Merge segments using the same logic as _quantize_mixed_batch
    # Determine token rate based on model name
    if model_name == "speech_tokenizer_v1":
        token_rate = 50
    else:
        token_rate = 25

    merged_codes = s3tokenizer.merge_tokenized_segments(
        segment_results, overlap=overlap, token_rate=token_rate
    )[:-overlap * token_rate]  # NOTE(xcsong): drop the last overlap part.
    return torch.tensor(merged_codes, dtype=torch.long)


def onnx_inference_with_long_audio_support(
        model_name: str, mel: torch.Tensor,
        mel_len: torch.Tensor) -> torch.Tensor:
    """
    ONNX inference with automatic long audio support
    """
    max_frames = 3000  # 30s * 16000 / 160 = 3000 frames

    if mel_len.item() <= max_frames:
        # Short audio - use direct inference
        return onnx_inference_short_audio(model_name, mel, mel_len)
    else:
        # Long audio - use sliding window approach
        return onnx_inference_long_audio(model_name, mel, mel_len)


def compare_torch_vs_onnx_single(model_name: str, audio: torch.Tensor,
                                 audio_name: str) -> Dict[str, Any]:
    """Test single audio with both torch and onnx versions"""
    duration = audio.shape[0] / 16000

    # Load torch model
    tokenizer = s3tokenizer.load_model(model_name)
    tokenizer.eval()

    # Prepare input
    mel = s3tokenizer.log_mel_spectrogram(audio)
    mels = mel.unsqueeze(0)
    mels_lens = torch.tensor([mel.size(1)])

    # Test torch version
    start_time = time.time()
    with torch.no_grad():
        torch_codes, torch_codes_lens = tokenizer.quantize(mels, mels_lens)
    torch_time = time.time() - start_time

    torch_result = torch_codes[0, :torch_codes_lens[0].item()]

    # Test onnx version with long audio support
    try:
        start_time = time.time()
        onnx_result = onnx_inference_with_long_audio_support(
            model_name, mel, mels_lens[0])
        onnx_time = time.time() - start_time

        # Compare results
        min_len = min(len(torch_result), len(onnx_result))
        torch_truncated = torch_result[:min_len]
        onnx_truncated = onnx_result[:min_len]

        are_equal = torch.equal(torch_truncated, onnx_truncated)
        miss_rate = 0.0

        if not are_equal:
            miss_num = torch.sum(~(torch_truncated == onnx_truncated))
            miss_rate = miss_num.item() * 100.0 / min_len

        return {
            "audio_name": audio_name,
            "model_name": model_name,
            "duration": duration,
            "torch_tokens": torch_truncated,
            "onnx_tokens": onnx_truncated,
            "torch_time": torch_time,
            "onnx_time": onnx_time,
            "results_match": are_equal,
            "miss_rate": miss_rate
        }

    except Exception as e:
        return {
            "audio_name": audio_name,
            "model_name": model_name,
            "duration": duration,
            "torch_tokens": torch_result,
            "onnx_tokens": [],
            "torch_time": torch_time,
            "onnx_time": 0.0,
            "results_match": False,
            "miss_rate": 100.0,
            "error": str(e)
        }


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1", "speech_tokenizer_v1_25hz",
    "speech_tokenizer_v2_25hz"
])
def test_torch_vs_onnx_short_audio(model_name, test_audio_suite):
    """Test torch vs onnx for short audio (<=30s)"""
    print(f"\n=== Testing {model_name} on Short Audio ===")

    short_audios = {
        k: v
        for k, v in test_audio_suite.items() if v.shape[0] / 16000 <= 30
    }

    results = []
    for audio_name, audio in short_audios.items():
        result = compare_torch_vs_onnx_single(model_name, audio, audio_name)
        results.append(result)

        duration = result["duration"]
        torch_tokens = result["torch_tokens"]
        onnx_tokens = result["onnx_tokens"]
        match_status = "✅" if result["results_match"] else "❌"

        print(
            f"{match_status} {audio_name}: {duration:.1f}s → torch:{len(torch_tokens)}, onnx:{len(onnx_tokens)}"
        )

        if not result["results_match"] and "error" not in result:
            print(f"   Miss rate: {result['miss_rate']:.2f}%")
            print(
                f"   torch_tokens:\n{torch_tokens}\nonnx_tokens:\n{onnx_tokens}"
            )

    # Assertions
    successful_tests = [r for r in results if "error" not in r]
    assert len(successful_tests) == len(
        short_audios
    ), f"successful tests ({len(successful_tests)}) for {model_name} should be equal to number of short audios ({len(short_audios)})"  # noqa

    # For short audio, we expect reasonable match rate
    for r in results:
        assert r[
            'miss_rate'] < 0.5, f"Miss rate too high for {model_name}: {r['miss_rate']:.2f}%"

    print(f"\n{model_name} Short Audio Summary:")
    print(f"  Successful tests: {len(successful_tests)}/{len(results)}")


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1", "speech_tokenizer_v1_25hz",
    "speech_tokenizer_v2_25hz"
])
def test_torch_vs_onnx_long_audio(model_name, test_audio_suite):
    """Test torch vs onnx for long audio (>30s) with ONNX sliding window implementation"""
    print(
        f"\n=== Testing {model_name} on Long Audio (ONNX Sliding Window) ===")

    long_audios = {
        k: v
        for k, v in test_audio_suite.items() if v.shape[0] / 16000 > 30
    }

    results = []
    for audio_name, audio in long_audios.items():
        result = compare_torch_vs_onnx_single(model_name, audio, audio_name)
        results.append(result)

        duration = result["duration"]
        torch_tokens = result["torch_tokens"]
        onnx_tokens = result["onnx_tokens"]
        match_status = "✅" if result["results_match"] else "❌"

        print(
            f"{match_status} {audio_name}: {duration:.1f}s → torch:{len(torch_tokens)}, onnx:{len(onnx_tokens)}"
        )

        if not result["results_match"] and "error" not in result:
            print(f"   Miss rate: {result['miss_rate']:.2f}%")
            print(
                f"   torch_tokens:\n{torch_tokens}\nonnx_tokens:\n{onnx_tokens}"
            )
        elif "error" in result:
            print(f"   Error: {result['error'][:100]}...")

    # For long audio with ONNX, we document the current limitations
    successful_tests = [r for r in results if "error" not in r]
    assert len(successful_tests) == len(
        long_audios
    ), f"successful tests ({len(successful_tests)}) for {model_name} should be equal to number of long audios ({len(long_audios)})"  # noqa

    print(f"\n{model_name} Long Audio Results:")
    print(f"  Total tests: {len(results)}")
    print(f"  Successful ONNX tests: {len(successful_tests)}")

    for r in results:
        # NOTE(xcsong): 0.5% is a reasonable miss rate for long audio, since we drop the last overlap part.
        assert r[
            'miss_rate'] < 0.5, f"Miss rate too high for {model_name}: {r['miss_rate']}%"

    # The main requirement is that Torch always works
    print("  ✅ Torch processing works reliably for all long audio")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
