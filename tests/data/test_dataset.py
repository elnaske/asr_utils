import pytest
from asr_utils.data import ASRDataset, asr_collate

import torch
import torchaudio
from torch.utils.data import DataLoader
import csv


@pytest.fixture
def audio_files(tmp_path):
    for i in range(3):
        torchaudio.save(tmp_path / f"{i}.wav",
                        torch.rand(1, 16000), sample_rate=16000)

    return [tmp_path / f"{i}.wav" for i in range(3)]


@pytest.fixture
def data_tsv(tmp_path, audio_files):
    with open(tmp_path / "data.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["path", "transcript", "etiology"])

        for wav in audio_files:
            writer.writerow([str(wav), "test", "test"])

    return tmp_path / "data.tsv"


@pytest.fixture
def data_tsv_no_meta(tmp_path, audio_files):
    with open(tmp_path / "data_no_meta.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["path", "transcript"])

        for wav in audio_files:
            writer.writerow([str(wav), "test"])

    return tmp_path / "data_no_meta.tsv"


@pytest.fixture
def data_tsv_invalid(tmp_path, audio_files):
    with open(tmp_path / "data_invalid.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["path"])

        for wav in audio_files:
            writer.writerow([str(wav)])

    return tmp_path / "data_invalid.tsv"


def test_asr_dataset(data_tsv):
    ds = ASRDataset(data_tsv)
    audio, ref, meta = ds[0]

    assert isinstance(audio, torch.Tensor)
    assert isinstance(ref, str)
    assert isinstance(meta, dict)
    assert "etiology" in meta.keys()


def test_asr_dataset_no_meta(data_tsv_no_meta):
    ds = ASRDataset(data_tsv_no_meta)
    audio, ref, meta = ds[0]

    assert isinstance(audio, torch.Tensor)
    assert isinstance(ref, str)
    assert isinstance(meta, dict)


def test_asr_dataset_invalid(data_tsv_invalid):
    with pytest.raises(RuntimeError):
        ASRDataset(data_tsv_invalid)


def test_asr_collate(data_tsv):
    ds = ASRDataset(data_tsv)
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

        assert "meta" in batch
        assert isinstance(batch["meta"], tuple)
        assert isinstance(batch["meta"][0], dict)

        assert batch["audio"].size(-1) == max(batch["ilens"])