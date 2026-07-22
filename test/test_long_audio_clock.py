import importlib.util
import math
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "s3tokenizer/chunking.py"
SPEC = importlib.util.spec_from_file_location("s3tokenizer_chunking_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
long_audio_frame_ranges = MODULE.long_audio_frame_ranges


def test_does_not_emit_redundant_sub_overlap_tail():
    total_frames = 5201
    ranges = long_audio_frame_ranges(
        total_frames, window_frames=3000, stride_frames=2600
    )
    assert ranges == [(0, 3000), (2600, 5201)]

    segment_token_counts = [math.ceil((end - start) / 4) for start, end in ranges]
    merged_token_count = sum(segment_token_counts) - 100  # One 4-second overlap at 25 Hz.
    assert merged_token_count == math.ceil(total_frames / 4)

    # The old loop added (5200, 5201). Half-overlap trimming discarded that
    # tiny terminal segment and another 50 tokens (exactly two seconds).
    old_segment_token_counts = segment_token_counts + [1]
    old_merged_token_count = sum(
        max(
            0,
            count
            - (50 if index else 0)
            - (50 if index != len(old_segment_token_counts) - 1 else 0),
        )
        for index, count in enumerate(old_segment_token_counts)
    )
    assert merged_token_count - old_merged_token_count == 50


@pytest.mark.parametrize(
    "total", (3001, 5000, 5199, 5200, 5201, 5210, 7799, 7800, 7801)
)
def test_boundary_lengths_end_exactly_once(total):
    ranges = long_audio_frame_ranges(
        total, window_frames=3000, stride_frames=2600
    )
    assert ranges[-1][1] == total
    assert sum(end == total for _, end in ranges) == 1
    assert all(end > start for start, end in ranges)


def test_validates_geometry():
    with pytest.raises(ValueError):
        long_audio_frame_ranges(10, window_frames=5, stride_frames=6)
