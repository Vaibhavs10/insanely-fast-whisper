from insanely_fast_whisper.utils.diarize import (
    diarize_audio,
    post_process_segments_and_transcripts,
)


class FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        yield from self._tracks


class FakePipeline:
    def __init__(self, tracks):
        self._tracks = tracks

    def __call__(self, inputs, num_speakers=None, min_speakers=None, max_speakers=None):
        return FakeAnnotation(self._tracks)


def test_diarize_audio_empty_diarization_returns_empty():
    pipeline = FakePipeline([])
    assert diarize_audio(None, pipeline, None, None, None) == []


def test_diarize_audio_merges_consecutive_same_speaker_segments():
    pipeline = FakePipeline(
        [
            (FakeSegment(0.0, 1.0), "t0", "SPEAKER_00"),
            (FakeSegment(1.0, 2.0), "t1", "SPEAKER_00"),
            (FakeSegment(2.0, 3.0), "t2", "SPEAKER_01"),
        ]
    )
    assert diarize_audio(None, pipeline, None, None, None) == [
        {"segment": {"start": 0.0, "end": 2.0}, "speaker": "SPEAKER_00"},
        {"segment": {"start": 2.0, "end": 3.0}, "speaker": "SPEAKER_01"},
    ]


def test_post_process_empty_transcript_returns_empty():
    segments = [{"segment": {"start": 0.0, "end": 1.0}, "speaker": "SPEAKER_00"}]
    assert post_process_segments_and_transcripts(segments, [], group_by_speaker=False) == []


def test_post_process_empty_segments_returns_empty():
    transcript = [{"text": " hi", "timestamp": (0.0, 1.0)}]
    assert post_process_segments_and_transcripts([], transcript, group_by_speaker=False) == []


def test_post_process_assigns_speakers_to_chunks():
    segments = [
        {"segment": {"start": 0.0, "end": 2.0}, "speaker": "SPEAKER_00"},
        {"segment": {"start": 2.0, "end": 4.0}, "speaker": "SPEAKER_01"},
    ]
    transcript = [
        {"text": " hello", "timestamp": (0.0, 2.0)},
        {"text": " world", "timestamp": (2.0, 4.0)},
    ]
    assert post_process_segments_and_transcripts(
        segments, transcript, group_by_speaker=False
    ) == [
        {"speaker": "SPEAKER_00", "text": " hello", "timestamp": (0.0, 2.0)},
        {"speaker": "SPEAKER_01", "text": " world", "timestamp": (2.0, 4.0)},
    ]


def test_post_process_handles_none_end_timestamp():
    segments = [{"segment": {"start": 0.0, "end": 2.0}, "speaker": "SPEAKER_00"}]
    transcript = [{"text": " hello", "timestamp": (0.0, None)}]
    assert post_process_segments_and_transcripts(
        segments, transcript, group_by_speaker=False
    ) == [{"speaker": "SPEAKER_00", "text": " hello", "timestamp": (0.0, None)}]
