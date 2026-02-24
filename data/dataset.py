import torch
from torch.utils.data import Dataset
from torchaudio import load
from torch.nn.utils.rnn import pad_sequence

class ASRDataset(Dataset):
    def __init__(self, paths_txt):
        with open(paths_txt, 'r') as f:
            lines = f.readlines()

        self.paths = [] # paths to audio files
        self.refs = []  # audio transcripts
        self.metas = [] # meta information (e.g. speaker identity)

        for line in lines:
            args = line.strip().split('\t')

            assert len(args) >= 2

            self.paths += [args[0]]
            self.refs += [args[1]]
            if len(args) > 2:
                self.metas += [tuple(args[2:])]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, sr = load(self.paths[idx])

        return audio, self.refs[idx], self.metas[idx] if self.metas else None
    

def asr_collate(batch):
    audios, refs, meta = zip(*batch)

    processed = []
    ilens = []

    for audio in audios:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # (1, T)
        processed += [audio.transpose(0, 1)] # (T, C)
        ilens += [audio.shape[-1]]

    padded = pad_sequence(processed, batch_first=True) # (B, T_max, C)
    padded = padded.transpose(1, 2) # (B, C, T_max)

    ilens = torch.tensor(ilens)

    return {
        "audio": padded,
        "ilens": ilens,
        "refs": refs,
        "meta": meta,
    }