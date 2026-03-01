import pytest
from asr_utils.data import ASRDataModule
from .test_dataset import audio_files, data_tsv
import torch
from torch.utils.data import DataLoader


def test_asr_data_module(data_tsv):
    data = ASRDataModule(
        train_tsv=data_tsv,
        val_tsv=data_tsv,
        batch_size=2,
    )
    data.setup(stage="fit")

    train_dl = data.train_dataloader()

    assert isinstance(train_dl, DataLoader)

    for batch in train_dl:
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