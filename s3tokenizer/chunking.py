"""Pure frame-window helpers used by long-audio tokenization."""

from __future__ import annotations


def long_audio_frame_ranges(
    total_frames: int,
    *,
    window_frames: int,
    stride_frames: int,
) -> list[tuple[int, int]]:
    """Cover an utterance without emitting a redundant terminal window."""
    if total_frames < 0 or window_frames <= 0 or stride_frames <= 0:
        raise ValueError("frame counts must be nonnegative and window/stride positive")
    if stride_frames > window_frames:
        raise ValueError("stride cannot exceed the window")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        ranges.append((start, end))
        if end >= total_frames:
            break
        start += stride_frames
    return ranges
