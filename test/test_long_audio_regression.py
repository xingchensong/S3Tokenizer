"""CPU regression tests for Issue #53; no weights or network are used.

From the checkout: python -m pytest -q test/test_long_audio_regression.py
Requires pytest and the normal s3tokenizer runtime dependencies.
The clock encoder is deliberately synthetic: these tests validate orchestration,
not learned token IDs, pretrained-model equivalence, or reconstruction quality.
"""

from pathlib import Path
import os
import sys

import pytest
import torch

# Also works when pytest is launched outside the checkout or without pip -e.
ROOT = Path(os.environ.get('S3TOKENIZER_REPO',
                          Path(__file__).resolve().parents[1]))
if not (ROOT / 's3tokenizer/model_v2.py').is_file():
    ROOT = Path.cwd()
assert (ROOT / 's3tokenizer/model_v2.py').is_file(), (
    'Run from the checkout or set S3TOKENIZER_REPO to its absolute path')
sys.path.insert(0, str(ROOT))

from s3tokenizer.model_v2 import AudioEncoderV2, S3TokenizerV2
from s3tokenizer.model_v3 import AudioEncoderV3, S3TokenizerV3
from s3tokenizer.utils import merge_tokenized_segments


WINDOW = 3000
STRIDE = 2600
INVALID = -999
BOUNDARIES = (
    1, 2, 3, 4, 5, 199, 200, 201,
    2599, 2600, 2601, 2999, 3000, 3001,
    5199, 5200, 5201, 5240, 5399, 5400, 5401,
    5599, 5600, 5601, 7799, 7800, 7801, 7840,
    7999, 8000, 8001, 8199, 8200, 8201, 10400, 10401,
)


def expected_ranges(length):
    # Independent closed-form oracle: minimum number of fixed-stride windows.
    count = 1 + max(0, (length - WINDOW + STRIDE - 1) // STRIDE)
    return [(i * STRIDE, min(i * STRIDE + WINDOW, length))
            for i in range(count)]


class ClockEncoder(torch.nn.Module):
    """Every four input frames represent one globally identifiable token."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, mel, lengths):
        self.calls.append((mel.clone(), lengths.clone()))
        valid = (lengths + 3) // 4
        hidden = mel[:, :1, ::4].transpose(1, 2).clone()
        for i, count in enumerate(valid.tolist()):
            hidden[i, count:] = INVALID
        return hidden, valid


class IdentityQuantizer(torch.nn.Module):
    def encode(self, hidden):
        return hidden.squeeze(-1).long()


def clock_model(version):
    cls = S3TokenizerV2 if version == 2 else S3TokenizerV3
    # Run the real quantize and _quantize_mixed_batch methods, without
    # allocating the large pretrained architecture or downloading a checkpoint.
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.encoder = ClockEncoder()
    model.quantizer = IdentityQuantizer()
    return model


def clock_inputs(lengths, extra_padding=0):
    mel = torch.full((len(lengths), 2, max(lengths) + extra_padding), -777.0)
    for i, length in enumerate(lengths):
        clock = 10000 * (i + 1) + torch.arange(length) // 4
        mel[i, :, :length] = clock
    return mel, torch.tensor(lengths, dtype=torch.int32)


def assert_clock(model, lengths, extra_padding=0):
    mel, mel_len = clock_inputs(lengths, extra_padding)
    codes, code_len = model.quantize(mel, mel_len)
    assert code_len.tolist() == [(length + 3) // 4 for length in lengths]
    expected_width = ((mel.shape[-1] + 3) // 4 if max(lengths) <= WINDOW
                      else max(code_len).item())
    assert codes.shape == (len(lengths), expected_width)
    for i, length in enumerate(code_len.tolist()):
        expected = 10000 * (i + 1) + torch.arange(length)
        assert torch.equal(codes[i, :length], expected)
        assert INVALID not in codes[i, :length].tolist()

    assert len(model.encoder.calls) == 1
    observed_mel, observed_len = model.encoder.calls[0]
    if max(lengths) <= WINDOW:
        # Existing fast path: all short inputs go directly to the encoder.
        assert torch.equal(observed_mel, mel)
        assert torch.equal(observed_len, mel_len)
        return codes, code_len

    ranges = [(i, start, end) for i, length in enumerate(lengths)
              for start, end in expected_ranges(length)]
    assert observed_mel.shape == (len(ranges), 2, WINDOW)
    assert observed_len.tolist() == [end - start for _, start, end in ranges]
    for j, (i, start, end) in enumerate(ranges):
        valid = end - start
        assert torch.equal(observed_mel[j, :, :valid], mel[i, :, start:end])
        assert torch.count_nonzero(observed_mel[j, :, valid:]).item() == 0
    # The unified mixed path constructs zero-padded output, independent of
    # the encoder's deliberately poisoned invalid token region.
    for i, valid in enumerate(code_len.tolist()):
        assert torch.count_nonzero(codes[i, valid:]).item() == 0
    return codes, code_len


@pytest.mark.parametrize('version', [2, 3])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
@pytest.mark.parametrize('frames', BOUNDARIES)
def test_clock_padding_lengths_and_batching(version, batch_size, frames):
    companions = [17, 3000, 5201, 7800, 5240, 3001, 1]
    lengths = [frames] + companions[:batch_size - 1]
    assert_clock(clock_model(version), lengths, extra_padding=13)


@pytest.mark.parametrize('version', [2, 3])
def test_same_utterance_alone_and_in_mixed_batches(version):
    # Keep sample index zero so that its synthetic token IDs are identical.
    reference, ref_len = assert_clock(clock_model(version), [5240])
    for companions in ([17], [7801, 3000, 1], [17] * 7):
        codes, lengths = assert_clock(clock_model(version), [5240] + companions)
        assert lengths[0] == ref_len[0]
        assert torch.equal(codes[0, :lengths[0]], reference[0])


def test_issue_53_with_actual_merge_function():
    # Literal windows from the old loop: [0,3000), [2600,5240), [5200,5240).
    old = [list(range(0, 750)), list(range(650, 1310)),
           list(range(1300, 1310))]
    truncated = merge_tokenized_segments(old, overlap=4, token_rate=25)
    assert truncated == list(range(1260))
    repaired = merge_tokenized_segments(old[:2], overlap=4, token_rate=25)
    assert repaired == list(range(1310))


@pytest.mark.parametrize('tail_frames', [1, 2, 3, 4, 5, 39, 40, 199, 200, 201, 399, 400])
def test_old_redundant_tail_loss_is_not_always_fifty(tail_frames):
    # At 52s + a tail, the old loop incorrectly added a third segment.
    total = 5200 + tail_frames
    segments = [list(range(start // 4, (end + 3) // 4))
                for start, end in ((0, 3000), (2600, total), (5200, total))]
    merged = merge_tokenized_segments(segments, overlap=4, token_rate=25)
    tail_tokens = (tail_frames + 3) // 4
    expected_loss = max(0, 50 - max(0, tail_tokens - 50))
    assert (total + 3) // 4 - len(merged) == expected_loss


@pytest.mark.parametrize('encoder_cls', [AudioEncoderV2, AudioEncoderV3])
@pytest.mark.parametrize('padded_frames', [3000, 3001])
def test_real_untrained_encoder_length_contract(encoder_cls, padded_frames):
    # Test actual convolution and length/mask arithmetic, without learned
    # attention blocks. This is not a pretrained-model or audio-quality test.
    encoder = encoder_cls(n_mels=2, n_state=4, n_head=1, n_layer=0,
                          stride=2, use_sdpa=False).eval()
    lengths = torch.tensor([1, 2, 3, 4, 5, 400, 401, 2601, 2999, 3000])
    mel = torch.zeros(len(lengths), 2, padded_frames)
    with torch.inference_mode():
        hidden, actual = encoder(mel, lengths)
    assert actual.tolist() == [(length + 3) // 4 for length in lengths.tolist()]
    assert hidden.shape == (len(lengths), (padded_frames + 3) // 4, 4)
    poisoned = mel.clone()
    for i, length in enumerate(lengths.tolist()):
        poisoned[i, :, length:] = 12345
    with torch.inference_mode():
        hidden_poisoned, actual_poisoned = encoder(poisoned, lengths)
    assert torch.equal(actual_poisoned, actual)
    assert torch.equal(hidden_poisoned, hidden)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
