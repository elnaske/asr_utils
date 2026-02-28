import torch
from torch.utils.data import Dataset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import csv
from typing import Union, Callable
from pathlib import Path


class ASRDataset(Dataset):
    def __init__(self, paths_tsv: Union[Path, str], load_fn: Callable = torchaudio.load):
        self.paths = []  # paths to audio files
        self.refs = []  # audio transcripts
        self.metas = []  # meta information (e.g. speaker identity)

        self.load_fn = load_fn

        with open(paths_tsv, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            headings = next(reader)
            n_col = len(headings)

            if n_col < 2:
                raise RuntimeError(
                    f"Error reading file {paths_tsv}: File has fewer than two columns, when it should have at least paths and transcripts")

            for row in reader:
                self.paths += [row[0]]
                self.refs += [row[1]]
                meta = {headings[i]: row[i] for i in range(2, n_col)} if n_col > 2 else {}
                self.metas += [meta]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, sr = self.load_fn(self.paths[idx])

        return audio, self.refs[idx], self.metas[idx]


def asr_collate(batch):
    audios, refs, meta = zip(*batch)

    processed = []
    ilens = []

    for audio in audios:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)
        processed += [audio.transpose(0, 1)]  # (T, C)
        ilens += [audio.shape[-1]]

    padded = pad_sequence(processed, batch_first=True)  # (B, T_max, C)
    padded = padded.transpose(1, 2)  # (B, C, T_max)

    ilens = torch.tensor(ilens)

    return {
        "audio": padded,
        "ilens": ilens,
        "refs": refs,
        "meta": meta,
    }
