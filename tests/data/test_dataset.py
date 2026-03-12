import pytest
from asr_utils.data import ASRDataset, asr_collate

import torch
import torchaudio
from torch.utils.data import DataLoader
import csv


@pytest.fixture
def audio_files(tmp_path):
    for i in range(3):
        torchaudio.save(tmp_path / f"{i}.wav", torch.rand(1, 16000), sample_rate=16000)

    return [tmp_path / f"{i}.wav" for i in range(3)]


@pytest.fixture
def data_csv(tmp_path, audio_files):
    with open(tmp_path / "data.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        # updated to match dataset layout
        writer.writerow(
            [
                "id",
                "speaker",
                "etiology",
                "audio_filepath",
                "duration",
                "text",
                "norm_text_with_disfluency",
                "norm_text_without_disfluency",
            ]
        )

        for wav in audio_files:
            writer.writerow(
                ["id", "speaker_id", "ALS", str(wav), 5.4, "test", "test", "tesssst"]
            )

    return tmp_path / "data.csv"


@pytest.fixture
def data_csv_no_meta(tmp_path, audio_files):
    with open(tmp_path / "data_no_meta.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["id", "audio_filepath", "norm_text_without_disfluency"])

        for wav in audio_files:
            writer.writerow(["id", str(wav), "test"])

    return tmp_path / "data_no_meta.csv"


@pytest.fixture
def data_csv_invalid(tmp_path, audio_files):
    with open(tmp_path / "data_invalid.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["path"])

        for wav in audio_files:
            writer.writerow([str(wav)])

    return tmp_path / "data_invalid.csv"


def test_asr_dataset(data_csv):
    ds = ASRDataset(data_csv)
    audio, ref, key, meta = ds[0]

    assert isinstance(audio, torch.Tensor)
    assert isinstance(ref, str)
    assert isinstance(key, str)
    assert isinstance(meta, dict)
    assert "etiology" in meta.keys()


def test_asr_dataset_no_meta(data_csv_no_meta):
    ds = ASRDataset(data_csv_no_meta)
    audio, ref, key, meta = ds[0]

    assert isinstance(audio, torch.Tensor)
    assert isinstance(ref, str)
    assert isinstance(key, str)
    assert isinstance(meta, dict)


def test_asr_dataset_invalid(data_csv_invalid):
    with pytest.raises(RuntimeError):
        ASRDataset(data_csv_invalid)


def test_asr_collate(data_csv):
    ds = ASRDataset(data_csv)
    dl = DataLoader(ds, 2, collate_fn=asr_collate)

    for batch in dl:
        assert isinstance(batch, dict)

        assert "audio" in batch
        assert isinstance(batch["audio"], torch.Tensor)

        assert "ilens" in batch
        assert isinstance(batch["ilens"], torch.Tensor)

        assert "refs" in batch
        assert isinstance(batch["refs"], tuple)
        assert isinstance(batch["refs"][0], str)

        assert "keys" in batch
        assert isinstance(batch["keys"], tuple)
        assert isinstance(batch["keys"][0], str)

        assert "meta" in batch
        assert isinstance(batch["meta"], tuple)
        assert isinstance(batch["meta"][0], dict)

        assert batch["audio"].size(-1) == max(batch["ilens"])
