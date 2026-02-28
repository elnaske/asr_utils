import pytest
from asr_utils.models import StreamingASRModel


class Model:
    def __init__(self):
        return

    def set_partial_callback(self, callback):
        pass

    def reset(self):
        pass

    def accept_chunk(self, audio_chunk):
        pass

    def input_finished(self):
        pass


class InvalidModel:
    def __init__(self):
        return

    def set_partial_callback(self, callback):
        pass


def test_streaming_asr_model():
    StreamingASRModel(Model, chunk_size=32)

    with pytest.raises(NotImplementedError):
        StreamingASRModel(InvalidModel, chunk_size=32)