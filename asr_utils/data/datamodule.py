import lightning as L
from asr_utils.data import ASRDataset, asr_collate
from torch.utils.data import DataLoader
import librosa
from typing import Callable

class ASRDataModule(L.LightningDataModule):
    def __init__(self, train_tsv: str, val_tsv: str, batch_size: int, load_fn: Callable = librosa.load):
        super().__init__()
        self.train_tsv = train_tsv
        self.val_tsv = val_tsv
        self.batch_size = batch_size
        self.load_fn = load_fn

    def setup(self, stage: str):
        self.train_set = ASRDataset(self.train_tsv, self.load_fn)
        self.val_set = ASRDataset(self.val_tsv, self.load_fn)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=asr_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=asr_collate)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["audio"] = batch["audio"].to(device)
        batch["ilens"] = batch["ilens"].to(device)
        return batch