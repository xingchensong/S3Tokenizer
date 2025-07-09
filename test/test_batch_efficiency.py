#!/usr/bin/env python3
"""
Batch processing efficiency test
Test the efficiency improvement of new batch processing functionality for mixed long and short audio
"""

import time
import torch
import pytest
import s3tokenizer


def create_test_audio(duration_seconds=20, sample_rate=16000):
    """Create test audio"""
    length = int(duration_seconds * sample_rate)
    # Create meaningful audio signal (sine wave mixture)
    t = torch.linspace(0, duration_seconds, length)
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)  # 440Hz fundamental
    audio += 0.3 * torch.sin(2 * torch.pi * 880 * t)  # 880Hz second harmonic
    audio += 0.1 * torch.randn(length)  # Add some noise
    return audio


@pytest.fixture
def test_audios():
    """Create test audio dataset"""
    return [
        create_test_audio(10),  # Short audio
        create_test_audio(20),  # Medium audio
        create_test_audio(40),  # Long audio
        create_test_audio(60),  # Long audio
        create_test_audio(15),  # Short audio
        create_test_audio(35),  # Long audio
        create_test_audio(25),  # Medium audio
        create_test_audio(50),  # Long audio
    ]


@pytest.fixture
def long_audios():
    """Create long audio dataset"""
    return [
        create_test_audio(45.5),
        create_test_audio(60),
        create_test_audio(91.2),
        create_test_audio(120),
    ]


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1_25hz", "speech_tokenizer_v1",
    "speech_tokenizer_v2_25hz"
])
def test_batch_efficiency(test_audios, model_name):
    """Test batch processing efficiency for different models"""
    print(f"\n=== Batch Processing Efficiency Test for {model_name} ===")

    # Load model
    model = s3tokenizer.load_model(model_name)
    model.eval()

    # Method 1: Individual processing
    print(f"\n--- Method 1: Individual Processing ({model_name}) ---")
    start_time = time.time()
    individual_results = []

    for i, audio in enumerate(test_audios):
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mels = mel.unsqueeze(0)
        mels_lens = torch.tensor([mel.size(1)])

        with torch.no_grad():
            codes, codes_lens = model.quantize(mels, mels_lens)

        final_codes = codes[0, :codes_lens[0].item()].tolist()
        individual_results.append(final_codes)

        duration = audio.shape[0] / 16000
        processing_type = "Long audio" if duration > 30 else "Short audio"
        print(
            f"Audio {i+1}: {duration:.1f}s, {len(final_codes)} tokens, {processing_type}"
        )

    individual_time = time.time() - start_time
    print(f"Individual processing total time: {individual_time:.2f}s")

    # Method 2: Batch processing
    print(f"\n--- Method 2: Batch Processing ({model_name}) ---")
    start_time = time.time()

    # Prepare batch input
    mels = []
    for audio in test_audios:
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mels.append(mel)

    # Use padding to handle different lengths of mel
    mels, mels_lens = s3tokenizer.padding(mels)

    # Batch processing
    with torch.no_grad():
        codes, codes_lens = model.quantize(mels, mels_lens)

    # Process results
    batch_results = []
    for i in range(len(test_audios)):
        final_codes = codes[i, :codes_lens[i].item()].tolist()
        batch_results.append(final_codes)

        duration = test_audios[i].shape[0] / 16000
        processing_type = "Long audio" if duration > 30 else "Short audio"
        print(
            f"Audio {i+1}: {duration:.1f}s, {len(final_codes)} tokens, {processing_type}"
        )

    batch_time = time.time() - start_time
    print(f"Batch processing total time: {batch_time:.2f}s")

    # Verify result consistency
    print(f"\n--- Result Verification for {model_name} ---")
    all_ok = True
    for i in range(len(test_audios)):
        individual_tokens = individual_results[i]
        batch_tokens = batch_results[i]

        # Calculate miss rate
        if len(individual_tokens) != len(batch_tokens):
            print(
                f"❌ Audio {i+1} length mismatch: individual={len(individual_tokens)}, batch={len(batch_tokens)}"
            )
            all_ok = False
        else:
            mismatches = sum(1 for a, b in zip(individual_tokens, batch_tokens)
                             if a != b)
            miss_rate = mismatches / len(individual_tokens) * 100 if len(
                individual_tokens) > 0 else 0

            if miss_rate < 0.2:  # Less than 0.2% is considered OK
                print(f"✅ Audio {i+1} miss rate: {miss_rate:.4f}% (OK)")
            else:
                print(f"❌ Audio {i+1} miss rate: {miss_rate:.4f}% (Too high)")
                all_ok = False

    # Efficiency improvement
    speedup = individual_time / batch_time
    print(f"\n--- Efficiency Improvement for {model_name} ---")
    print(f"Batch processing speedup: {speedup:.2f}x")
    if speedup > 1:
        print("✅ Batch processing indeed improves efficiency!")
    else:
        print("⚠️  Batch processing doesn't significantly improve efficiency")

    # Assertions for pytest
    assert all_ok, f"Results don't match for model {model_name}"
    assert len(individual_results) == len(
        batch_results), "Number of results don't match"
    assert all(
        len(individual_results[i]) == len(batch_results[i])
        for i in range(len(test_audios))), "Token counts don't match"

    # Performance assertion - batch should be at least as fast as individual (allowing for some variance)
    # assert batch_time <= individual_time * 1.1, f"Batch processing should not be significantly slower than individual processing for {model_name}"


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1_25hz", "speech_tokenizer_v1",
    "speech_tokenizer_v2_25hz"
])
def test_pure_long_audio_batch(long_audios, model_name):
    """Test pure long audio batch processing for different models"""
    print(f"\n=== Pure Long Audio Batch Processing Test for {model_name} ===")

    model = s3tokenizer.load_model(model_name)
    model.eval()

    # Prepare batch input
    mels = []
    for audio in long_audios:
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mels.append(mel)

    mels, mels_lens = s3tokenizer.padding(mels)

    # Batch process long audio
    start_time = time.time()
    with torch.no_grad():
        codes, codes_lens = model.quantize(mels, mels_lens)
    processing_time = time.time() - start_time

    print(
        f"Batch processing {len(long_audios)} long audios took: {processing_time:.2f}s"
    )

    results = []
    for i in range(len(long_audios)):
        duration = long_audios[i].shape[0] / 16000
        tokens_count = codes_lens[i].item()
        results.append((duration, tokens_count))
        print(f"Long audio {i+1}: {duration:.1f}s → {tokens_count} tokens")

    print(
        f"✅ Pure long audio batch processing test completed for {model_name}")

    # Assertions for pytest
    assert codes is not None, f"Codes should not be None for model {model_name}"
    assert codes_lens is not None, f"Codes lengths should not be None for model {model_name}"
    assert len(results) == len(
        long_audios), "Number of results should match number of input audios"
    assert all(
        tokens_count > 0
        for _, tokens_count in results), "All audio should produce tokens"
    assert processing_time > 0, "Processing time should be positive"


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1_25hz", "speech_tokenizer_v1",
    "speech_tokenizer_v2_25hz"
])
def test_model_loading(model_name):
    """Test that all models can be loaded successfully"""
    print(f"\n=== Model Loading Test for {model_name} ===")

    model = s3tokenizer.load_model(model_name)
    assert model is not None, f"Model {model_name} should load successfully"

    # Test model can be set to eval mode
    model.eval()
    print(f"✅ Model {model_name} loaded and set to eval mode successfully")


@pytest.mark.parametrize("model_name", [
    "speech_tokenizer_v1_25hz", "speech_tokenizer_v1",
    "speech_tokenizer_v2_25hz"
])
def test_single_audio_processing(model_name):
    """Test single audio processing for different models"""
    print(f"\n=== Single Audio Processing Test for {model_name} ===")

    # Create a single test audio
    audio = create_test_audio(30)  # 30 second audio

    model = s3tokenizer.load_model(model_name)
    model.eval()

    # Process the audio
    mel = s3tokenizer.log_mel_spectrogram(audio)
    mels = mel.unsqueeze(0)
    mels_lens = torch.tensor([mel.size(1)])

    with torch.no_grad():
        codes, codes_lens = model.quantize(mels, mels_lens)

    final_codes = codes[0, :codes_lens[0].item()].tolist()

    # Assertions
    assert codes is not None, f"Codes should not be None for model {model_name}"
    assert codes_lens is not None, f"Codes lengths should not be None for model {model_name}"
    assert len(
        final_codes) > 0, f"Should produce tokens for model {model_name}"
    assert codes_lens[0].item() == len(
        final_codes
    ), f"Codes length should match actual codes for model {model_name}"

    duration = audio.shape[0] / 16000
    print(
        f"✅ Single audio processing test completed for {model_name}: {duration:.1f}s → {len(final_codes)} tokens"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
