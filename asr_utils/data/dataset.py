import torch
from torch.utils.data import Dataset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import csv
from typing import Union, Callable
from pathlib import Path


class ASRDataset(Dataset):
    def __init__(
        self, paths_csv: Union[Path, str], load_fn: Callable = torchaudio.load
    ):
        self.paths = []  # paths to audio files
        self.refs = []  # audio transcripts
        self.ids = []  # utterance ids, need for their evaluation
        self.metas = []  # meta information (e.g. speaker identity)

        self.load_fn = load_fn

        with open(paths_csv, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            headings = next(reader)
            n_col = len(headings)

            if n_col < 3:
                raise RuntimeError(
                    f"Error reading file {paths_csv}: File has fewer than three columns, when it should have at least paths, ids, and transcripts"
                )

            for row in reader:
                # adjusted to include full file path.
                self.paths.append(row["audio_filepath"])
                self.ids.append(row["id"])
                self.refs.append(row["norm_text_without_disfluency"])
                # need to handle this section better
                meta = {
                    k: v
                    for k, v in row.items()
                    if k not in ["audio_filepath", "id", "norm_text_without_disfluency"]
                }
                self.metas += [meta]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, sr = self.load_fn(self.paths[idx])

        # considering adding return of self.ids[idx]
        return audio, self.refs[idx], self.paths[idx], self.metas[idx]


def asr_collate(batch):
    audios, refs, keys, meta = zip(*batch)

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
        "keys": keys,
        "meta": meta,
    }
